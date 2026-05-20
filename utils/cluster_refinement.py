import json
import os
import hashlib
import random
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from utils.env import load_dotenv_file
from utils.output_paths import OutputPaths
from threadpoolctl import threadpool_limits

# 决策置信度等级，数值越大保护性越强：
#   fallback : 三次调用两两不同 / 未审查 / 超时回退 —— 最该被数量归一调整
#   weak     : 三次里两次相同一次不同（弱共识）—— 次优先被调整
#   strong   : 三次完全一致（强共识 keep / split 强证据产物）—— 数量归一绝不触碰
_CONF_RANK = {"fallback": 0, "weak": 1, "strong": 2}


class ClusterPacketBuilder:
    """构建供离线/外部 LLM 使用的未知簇信息包。"""

    def __init__(self, topk_representatives=8):
        """初始化未知簇信息包的构建参数。"""
        self.topk_representatives = topk_representatives

    @staticmethod
    def _is_valid_utterance(text_a):
        """判断当前句（text_a）是否有实质内容，过滤以下情况：
        1. 空字符串或纯空白
        2. 纯数字（单个或多个连续数字）
        
        注：不再按 token 数过滤，因为有上下文补充。
        """
        if not isinstance(text_a, str):
            return False
        stripped = text_a.strip()
        if not stripped:
            return False
        if stripped.isdigit():
            return False
        return True

    def build(self, features, examples, assignments, centroids):
        """根据聚类结果构建簇级信息包。

        新结构：每个 cluster 包含一个 examples 数组，每个 example 包含：
          - utterance: 当前句（带 speaker 标记）
          - context: 三句拼接的上下文（[PREV] | [CURR] | [NEXT]）

        Args:
            examples: InputExample 对象列表，每条含 text_a / text_pre / text_next
                      及对应 speaker 字段；或字符串列表（向后兼容，仅用 text_a）。
        """
        packets = []
        num_clusters = centroids.shape[0]
        norm_features = F.normalize(features, dim=1)
        norm_centroids = F.normalize(centroids, dim=1)
        sim_matrix = torch.matmul(norm_features, norm_centroids.T)

        # 兼容旧式字符串列表调用
        _is_example = examples and not isinstance(examples[0], str)

        def _get_text_a(idx):
            if _is_example:
                ex = examples[idx]
                spk = getattr(ex, "speaker", "") or ""
                tag = f"[{spk.upper()}] " if spk else ""
                return tag + ex.text_a
            return examples[idx]

        def _build_context(idx):
            """构造三句拼接的上下文字符串，供 LLM 理解当前句的话语功能。"""
            if not _is_example:
                return None
            ex = examples[idx]

            def _fmt(speaker, text):
                if not text:
                    return None
                spk = (speaker or "").strip().upper()
                tag = f"[{spk}] " if spk else ""
                return tag + text.strip()

            pre  = _fmt(getattr(ex, "speaker_pre",  None), getattr(ex, "text_pre",  None))
            curr = _fmt(getattr(ex, "speaker",       None), getattr(ex, "text_a",    None))
            nxt  = _fmt(getattr(ex, "speaker_next",  None), getattr(ex, "text_next", None))

            parts = []
            if pre:
                parts.append(f"[PREV] {pre}")
            if curr:
                parts.append(f"[CURR] {curr}")
            if nxt:
                parts.append(f"[NEXT] {nxt}")
            return " | ".join(parts) if parts else None

        for cluster_id in range(num_clusters):
            member_mask = assignments == cluster_id
            member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
            if member_indices.numel() == 0:
                continue

            cluster_sims = sim_matrix[member_indices, cluster_id]
            rep_order = torch.argsort(cluster_sims, descending=True)

            # 按相似度排序后过滤无效 utterance，取前 topk 个有效项
            def _pick_valid(order, topk):
                result = []
                for pos in order.cpu().tolist():
                    idx = member_indices[pos].item()
                    if idx < len(examples):
                        # 只检查当前句的有效性
                        if _is_example:
                            text_a = examples[idx].text_a
                        else:
                            text_a = examples[idx]
                        
                        if self._is_valid_utterance(text_a):
                            utterance = _get_text_a(idx)
                            context = _build_context(idx)
                            result.append({
                                "utterance": utterance,
                                "context": context
                            })
                            if len(result) >= topk:
                                break
                return result

            rep_examples = _pick_valid(rep_order, self.topk_representatives)

            packet = {
                "cluster_id": int(cluster_id),
                "representative": rep_examples,
            }

            packets.append(packet)

        return packets


class ClusterRefiner:
    """应用在线/离线 LLM 决策并保持未知簇数量固定。"""

    # Talk Moves 类别定义文件路径（相对于项目根目录）
    TALK_MOVES_CLASSES_PATH = "data/talkmoves/talkmoves_classes.json"

    def __init__(self, args, known_label_list=None, epoch=None):
        """初始化簇修正器及其外部配置。

        Args:
            args: 命令行参数对象
            known_label_list: 已知类别标签列表，用于提取对应的类别定义信息
            epoch: 当前训练轮次，用于文件名区分
        """
        self.args = args
        self.known_label_list = known_label_list or []
        self.epoch = epoch
        # 真实的未知类数量：总类别数 - 已知类数，在训练开始前就已确定固定，
        # 不受训练过程中 KMeans/LLM merge-split 的影响。
        total_classes = len(getattr(args, 'id2label', {}))
        self._true_unknown_count = max(0, total_classes - len(self.known_label_list))
        load_dotenv_file(args.llm_env_file)
        self.output_paths = OutputPaths(args)
        self.packet_builder = ClusterPacketBuilder(
            topk_representatives=args.packet_topk_representatives,
        )
        # 加载类别定义
        self._class_definitions = self._load_class_definitions()

    def _load_class_definitions(self):
        """加载 Talk Moves 类别定义文件。"""
        import os as _os
        # 尝试多种路径：项目根目录、data 目录、当前工作目录
        possible_paths = [
            self.TALK_MOVES_CLASSES_PATH,
            _os.path.join(_os.path.dirname(__file__), "..", "..", self.TALK_MOVES_CLASSES_PATH),
            _os.path.join(_os.getcwd(), self.TALK_MOVES_CLASSES_PATH),
        ]
        for path in possible_paths:
            if _os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data.get("classes", [])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load class definitions from {path}: {e}")
                    return []
        print(f"Warning: Class definition file not found in any of: {possible_paths}")
        return []

    def _get_known_class_info(self):
        if not self.known_label_list or not self._class_definitions:
            return []
        selected = []
        for label in self.known_label_list[:3]:
            for cls_def in self._class_definitions:
                if str(cls_def.get("id")) == str(label) or cls_def.get("name") == label:
                    selected.append({
                        "name": cls_def.get("name", ""),
                        "examples": cls_def.get("examples", [])[:3],  # 示例每类取3个示例
                    })
                    break
        return selected

    def _get_training_progress(self):
        """返回当前训练进度信息（同时提供 1-based 与 0-based epoch）。"""
        total_epochs = int(self.args.num_train_epochs)
        raw_epoch = int(self.epoch) if isinstance(self.epoch, int) else -1
        current_epoch = raw_epoch + 1 if raw_epoch >= 0 else 0
        return {
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "current_epoch_index": raw_epoch,
        }

    def attach_training_progress_to_packets(self, packets):
        """已弃用：training_progress 现在只在文件顶层写一次，不再写入每个 cluster packet。
        保留此方法以兼容现有调用点，直接原样返回 packets。
        """
        return packets

    def compute_packet_fingerprint(self, packets):
        """计算信息包集合的内容指纹。"""
        canonical = json.dumps(packets, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def build_system_prompt(self, known_class_info=None, num_clusters=None):
        """构建在线 LLM 修正使用的系统提示词。

        Args:
            known_class_info: 已知类信息列表（用于参考）。
            num_clusters: 当前未知簇的实际数量，用于约束 LLM 的 merge/split 决策。
        """
        known_context = ""
        if known_class_info:
            known_context = (
                "\n\nFor reference only — the following are KNOWN discourse categories already labeled in the dataset. "
                "Do NOT assign any cluster to these categories; they are provided solely to help you understand what discourse functions have already been identified, so you can focus on discovering NEW ones.\n"
            )
            for cls in known_class_info:
                examples_str = "; ".join(cls.get("examples", [])[:3]) if cls.get("examples") else "N/A"
                known_context += f"- {cls.get('name', 'Unknown')}: {examples_str}\n"

        base_prompt = (
            "System Prompt:\n"
            "You are an expert in classroom discourse analysis, responsible for refining the clustering results of teacher and student utterances from mathematics classrooms.\n"
            "\n"
            "Core Rules:\n"
            "1. Each cluster must correspond to exactly ONE coherent discourse function; clusters that fail to meet this requirement must be corrected via 'merge' or 'split'.\n"
            "2. For all utterances provided to you, the 'utterance' field refers to the current utterance, and the 'context' field indicates the position of the current utterance in the dialogue context. You must analyze the utterance based on its discourse function in context. The same utterance may have different discourse functions in different contexts.\n"
            "3. '[s]' denotes a student utterance and '[t]' denotes a teacher utterance.\n"
            "\n"
            "Input Field Definitions:\n"
            "1. cluster_id: Cluster identifier.\n"
            "2. representative-utterance: The samples closest to the cluster center with the highest typicality.\n"
            "3. representative-context: The representative utterance concatenated with its preceding and following utterances.\n"
            "\n"
            "Decision Rules:\n"
            "1. keep: The cluster corresponds to only one clear discourse function and has distinct functional differences from neighboring clusters.\n"
            "2. merge: Two clusters correspond to identical or highly similar discourse functions.\n"
            "3. split: The cluster clearly mixes two or more distinct discourse functions.\n"
            "\n"
            "Decision Basis:\n"
            "1. If the 'representative utterances' of a cluster obviously contain more than one distinct discourse function, this shall be regarded as strong evidence for 'split'.\n"
            "2. Compare 'representative utterances' across all other clusters; consistent discourse functions of 'representative utterances' shall be regarded as strong evidence for 'merge'.\n"
            "3. If neither of the above two conditions is satisfied, set the decision to 'keep'.\n"
            "\n"
            "Decision Requirements:\n"
            "1. Judgments based merely on superficial wording are prohibited; decisions must focus on the underlying discourse function.\n"
            "2. Different speaker roles can be taken as a reference for discourse function judgment, but 'split' cannot be adopted merely because of different speakers.\n"
            f"3. Total number of clusters no less than {num_clusters}.\n"
            "4. Defaulting to an all-keep decision is prohibited. The all-keep solution is valid when every cluster features unified internal discourse functions and clear functional differentiation from neighboring clusters.\n"
        )    
        return base_prompt + known_context

    def build_user_payload(self, packets):
        """构建发送给 LLM 的用户负载。"""
        progress = self._get_training_progress()
        return {
            "task": "cluster_level_refinement",
            "training_progress": progress,
            "instructions": {
                "allowed_decisions": ["keep", "merge", "split"],
                "required_keys": ["cluster_id", "decision", "target_cluster_id", "summary"],
                "notes": [
                    "Operate at the cluster level only.",
                    "Use target_cluster_id only for merge; set to null for keep and split.",
                    "If the decision is 'merge', the 'decision' field of both clusters must be set to 'merge', and the 'target_cluster_id' must be marked with the other's cluster_id.",
                    "The 'summary' field must explicitly state the strongest supporting evidence for this decision (not vague wording).",
                    "Return a JSON array only. No text outside the JSON."
                ]
            },
            "clusters": packets,
        }
    
    def shuffle_packets_for_attempt(self, packets, attempt_idx):
        """为单次在线调用打乱 packet 内例子顺序（仅打乱，不改字段）。"""
        epoch_offset = int(self.epoch) if isinstance(self.epoch, int) else -1
        rng = random.Random(int(self.args.seed) * 1009 + attempt_idx * 7919 + epoch_offset * 17)
        shuffled_packets = []
        for packet in packets:
            rep = list(packet.get("representative", []))
            rng.shuffle(rep)

            copied = {
                "cluster_id": packet.get("cluster_id"),
                "representative": rep,
            }
            shuffled_packets.append(copied)
        return shuffled_packets

    @staticmethod
    def has_strong_evidence_summary(summary):
        """校验 summary 是否明确写出强依据。"""
        if not isinstance(summary, str):
            return False
        text = summary.strip()
        if len(text) < 12:
            return False
        evidence_markers = [
            "依据", "强依据", "证据", "because", "based on", "evidence",
            "representative", "代表",
        ]
        text_lower = text.lower()
        return any(marker in text or marker in text_lower for marker in evidence_markers)

    def resolve_api_key(self):
        """解析在线 LLM 所需的 API Key。"""
        if self.args.llm_key:
            return self.args.llm_key
        if self.args.llm_key_env:
            return os.environ.get(self.args.llm_key_env)
        return None

    def resolve_api_url(self):
        """解析在线 LLM 的请求地址。"""
        if self.args.llm_url:
            return self.args.llm_url
        if self.args.llm_url_env:
            return os.environ.get(self.args.llm_url_env)
        return None

    def resolve_api_model(self):
        """解析在线 LLM 的模型名称。"""
        if self.args.llm_model:
            return self.args.llm_model
        if self.args.llm_model_env:
            return os.environ.get(self.args.llm_model_env)
        return None

    def parse_extra_headers(self):
        """解析额外的 HTTP 请求头配置。"""
        if not self.args.llm_headers:
            return {}
        try:
            payload = json.loads(self.args.llm_headers)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in --llm_headers: {exc}")
        if not isinstance(payload, dict):
            raise ValueError("--llm_headers must decode to a JSON object.")
        return {str(key): str(value) for key, value in payload.items()}

    def build_online_request_body(self, packets, packet_fingerprint):
        """构建在线 LLM 请求体。"""
        known_class_info = self._get_known_class_info()
        progress = self._get_training_progress()
        return {
            "model": self.resolve_api_model(),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.build_system_prompt(known_class_info, num_clusters=self._true_unknown_count)},
                {"role": "user", "content": json.dumps(self.build_user_payload(packets), ensure_ascii=False)},
            ],
            "metadata": {
                "packet_fingerprint": packet_fingerprint,
                "dataset": self.args.internal_dataset,
                "seed": self.args.seed,
                "current_epoch": progress["current_epoch"],
                "total_epochs": progress["total_epochs"],
            },
        }

    def extract_decisions_from_response(self, response_payload):
        """从接口响应中提取决策列表。"""
        if isinstance(response_payload, list):
            return response_payload

        if isinstance(response_payload, dict):
            if isinstance(response_payload.get("decisions"), list):
                return response_payload["decisions"]

            choices = response_payload.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = "".join(text_parts)
                if isinstance(content, str):
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
                        return parsed["decisions"]
                    if isinstance(parsed, list):
                        return parsed

        raise ValueError("Unable to parse LLM API response into a decisions list.")

    def validate_decisions(self, decisions, max_clusters):
        """校验并规范化簇级决策结果。"""
        valid_decisions = []
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            cluster_id = decision.get("cluster_id")
            action = decision.get("decision")
            target_cluster_id = decision.get("target_cluster_id")
            if not isinstance(cluster_id, int) or not (0 <= cluster_id < max_clusters):
                continue
            if action not in {"keep", "merge", "split"}:
                continue
            if action == "merge" and (not isinstance(target_cluster_id, int) or not (0 <= target_cluster_id < max_clusters)):
                continue
            summary = decision.get("summary")
            if not self.has_strong_evidence_summary(summary):
                continue
            # 透传置信度等级（缺省按 weak 处理，避免被当作 strong 锁定）
            confidence = decision.get("confidence", "weak")
            if confidence not in _CONF_RANK:
                confidence = "weak"
            normalized = {
                "cluster_id": cluster_id,
                "decision": action,
                "target_cluster_id": target_cluster_id if action == "merge" else None,
                "summary": summary,
                "confidence": confidence,
            }
            valid_decisions.append(normalized)
        return valid_decisions

    @staticmethod
    def _pick_longest_summary(items):
        candidates = [item.get("summary") for item in items if isinstance(item.get("summary"), str)]
        if not candidates:
            return None
        return max(candidates, key=lambda s: len(s))

    def majority_vote_decisions(self, run_decisions_list, packets):
        """三次在线决策投票：两次相同取之，三次全异则 keep。

        投票同时确定置信度等级：
          - 三票完全一致 -> strong（数量归一时锁定，绝不触碰）
          - 两票相同一票异 -> weak（次优先被数量归一调整）
          - 三票两两不同回退 keep -> fallback（最优先被数量归一调整）
        """
        packet_cluster_ids = [int(packet.get("cluster_id")) for packet in packets if isinstance(packet.get("cluster_id"), int)]
        decision_maps = []
        for decisions in run_decisions_list:
            decision_map = {}
            for item in decisions:
                cid = item.get("cluster_id")
                if isinstance(cid, int):
                    decision_map[cid] = item
            decision_maps.append(decision_map)

        final_decisions = []
        for cid in packet_cluster_ids:
            votes = []
            for decision_map in decision_maps:
                vote = decision_map.get(cid)
                if vote is None:
                    vote = {
                        "cluster_id": cid,
                        "decision": "keep",
                        "target_cluster_id": None,
                        "summary": "强依据：该轮未返回可用决策，按规则回退为 keep。",
                    }
                votes.append(vote)

            # “决策相同”按完整决策键比较：
            # keep/split -> (decision, None)
            # merge      -> (decision, target_cluster_id)
            decision_key_counts = {}
            key_to_votes = {}
            for vote in votes:
                action = vote.get("decision", "keep")
                target = vote.get("target_cluster_id") if action == "merge" else None
                decision_key = (action, target)
                decision_key_counts[decision_key] = decision_key_counts.get(decision_key, 0) + 1
                key_to_votes.setdefault(decision_key, []).append(vote)

            chosen_key = None
            chosen_count = 0
            for decision_key, count in decision_key_counts.items():
                if count >= 2:
                    chosen_key = decision_key
                    chosen_count = count
                    break

            if chosen_key is None:
                # 三次两两不同：回退 keep，标记为 fallback（残余簇，最优先被数量归一调整）
                final_decisions.append({
                    "cluster_id": cid,
                    "decision": "keep",
                    "target_cluster_id": None,
                    "summary": "强依据：三次调用决策两两不同，未形成两次相同决策，按规则保持 keep。",
                    "confidence": "fallback",
                })
                continue

            chosen_action, chosen_target = chosen_key
            winner_votes = key_to_votes.get(chosen_key, [])
            # 三票全同 -> strong（锁定）；两票相同一票异 -> weak
            conf_level = "strong" if chosen_count >= 3 else "weak"
            final_decisions.append({
                "cluster_id": cid,
                "decision": chosen_action,
                "target_cluster_id": chosen_target if chosen_action == "merge" else None,
                "summary": self._pick_longest_summary(winner_votes) or "强依据：该决策在三次调用中至少出现两次，按多数票采纳。",
                "confidence": conf_level,
            })

        return final_decisions

    def fetch_online_decisions_once(self, packets, packet_fingerprint):
        """单次在线调用，返回规范化后的决策结果。"""
        api_url = self.resolve_api_url()
        if not api_url:
            raise ValueError("Online LLM refinement requires --llm_url or LLM_API_URL in .env/environment.")
        headers = {
            "Content-Type": "application/json",
        }
        api_key = self.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.parse_extra_headers())

        body = json.dumps(self.build_online_request_body(packets, packet_fingerprint), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            api_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.llm_timeout) as response:
                raw_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM API HTTP error {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM API request failed: {exc}") from exc

        parsed_payload = json.loads(raw_payload)
        decisions = self.extract_decisions_from_response(parsed_payload)
        return self.validate_decisions(decisions, max_clusters=len(packets))

    def write_online_decisions(self, decisions, packet_fingerprint):
        """将在线决策写入可回放文件。"""
        if not decisions:
            return None
        path = self.output_paths.llm_decisions_write_path(epoch=self.epoch)
        if not path:
            return None
        self.output_paths.ensure_parent_dir(path)
        progress = self._get_training_progress()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({
                "packet_fingerprint": packet_fingerprint,
                "dataset": self.args.internal_dataset,
                "seed": self.args.seed,
                "epoch": self.epoch,
                "current_epoch": progress["current_epoch"],
                "total_epochs": progress["total_epochs"],
                "decisions": decisions,
            }, handle, indent=2, ensure_ascii=False)
        return path

    def fetch_online_decisions(self, packets, packet_fingerprint):
        """调用在线接口获取簇修正决策。"""
        run_decisions = []
        for attempt_idx in range(3):
            shuffled_packets = self.shuffle_packets_for_attempt(packets, attempt_idx)
            run_decisions.append(self.fetch_online_decisions_once(shuffled_packets, packet_fingerprint))
        final_decisions = self.majority_vote_decisions(run_decisions, packets)
        self.write_online_decisions(final_decisions, packet_fingerprint)
        return final_decisions

    def dump_packets(self, packets):
        """将簇信息包导出到本地文件。"""
        if not packets:
            return None, None, packets
        dump_path = self.output_paths.cluster_packets_path(epoch=self.epoch)
        self.output_paths.ensure_parent_dir(dump_path)
        packet_fingerprint = self.compute_packet_fingerprint(packets)
        progress = self._get_training_progress()
        with open(dump_path, "w", encoding="utf-8") as handle:
            json.dump({
                "packet_fingerprint": packet_fingerprint,
                "dataset": self.args.internal_dataset,
                "seed": self.args.seed,
                "training_progress": progress,
                "clusters": packets,
            }, handle, indent=2, ensure_ascii=False)
        return dump_path, packet_fingerprint, packets

    def load_decisions(self, expected_packet_fingerprint=None):
        """从本地文件加载离线决策。"""
        path = self.args.llm_decisions
        if path is None or not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            packet_fingerprint = payload.get("packet_fingerprint")
            if expected_packet_fingerprint and packet_fingerprint and packet_fingerprint != expected_packet_fingerprint:
                raise ValueError(
                    "Replay decisions fingerprint does not match the current cluster packets. "
                    "Regenerate decisions for this packet set or disable replay fallback."
                )
            return payload.get("decisions", [])
        if isinstance(payload, list):
            if expected_packet_fingerprint:
                raise ValueError(
                    "Replay decisions file is missing packet_fingerprint metadata. "
                    "Please regenerate it with the new format."
                )
            return payload
        return []

    def _make_keep_decisions(self, packets):
        """为所有簇生成 keep 决策，用于超时等异常情况的兜底。

        兜底 keep 全部标记为 fallback：这些 keep 并非语义判定结果，
        应优先被数量归一调整，不能享受 strong 锁定保护。
        """
        return [
            {"cluster_id": p["cluster_id"], "decision": "keep",
             "target_cluster_id": None, "summary": None, "confidence": "fallback"}
            for p in packets
        ]

    # ────────────────────────────────────────────────────────────────────
    # 消融对照：随机 / 几何决策生成（不经过 LLM）
    # ────────────────────────────────────────────────────────────────────
    def _random_decisions(self, packets):
        """+Random Refine：对每个簇随机产生 keep/merge/split 决策。

        用于排除“任何重组都有效”的可能性。所有决策标记为 weak，
        因此不享受 strong 锁定保护，数量归一可自由调整。
        """
        epoch_offset = int(self.epoch) if isinstance(self.epoch, int) else -1
        rng = random.Random(int(self.args.seed) * 131 + epoch_offset * 7 + 3)
        cluster_ids = [int(p["cluster_id"]) for p in packets if isinstance(p.get("cluster_id"), int)]
        decisions = []
        for cid in cluster_ids:
            # 轻微偏向 keep，避免每轮把结构打得过碎
            action = rng.choice(["keep", "keep", "merge", "split"])
            target = None
            if action == "merge":
                others = [c for c in cluster_ids if c != cid]
                if others:
                    target = rng.choice(others)
                else:
                    action = "keep"
            decisions.append({
                "cluster_id": cid,
                "decision": action,
                "target_cluster_id": target,
                "summary": "random refinement decision (ablation, no semantic basis).",
                "confidence": "weak",
            })
        return decisions

    def _geometric_decisions(self, features, assignments, centroids,
                             split_var_quantile=0.75, merge_sim_thresh=0.85):
        """+Geometric Refine：仅依据特征几何（簇内方差 / 簇间相似度）做 merge/split。

        用于证明 LLM 的话语功能级语义判断优于纯几何判断。
        所有决策标记为 weak（无语义依据），不享受 strong 锁定保护。
        """
        num_clusters = centroids.size(0)

        # 1. 每簇内部方差
        variances = []
        for cid in range(num_clusters):
            member_mask = assignments == cid
            member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
            if member_indices.numel() < 4:
                variances.append(-1.0)
                continue
            feats = features[member_indices]
            var = torch.mean((feats - centroids[cid].unsqueeze(0)).pow(2)).item()
            variances.append(var)

        valid_vars = sorted(v for v in variances if v >= 0)
        if valid_vars:
            idx = min(len(valid_vars) - 1, int(len(valid_vars) * split_var_quantile))
            split_thresh = valid_vars[idx]
        else:
            split_thresh = float("inf")

        # 2. 簇间质心余弦相似度
        norm_c = F.normalize(centroids, dim=1)
        sim = torch.matmul(norm_c, norm_c.T)
        sim.fill_diagonal_(-1.0)

        decisions = []
        merged = set()
        for cid in range(num_clusters):
            if cid in merged:
                decisions.append({
                    "cluster_id": cid, "decision": "keep", "target_cluster_id": None,
                    "summary": "geometric: already merged with a more similar cluster.",
                    "confidence": "weak",
                })
                continue
            row = sim[cid]
            best_j = int(torch.argmax(row).item())
            best_sim = float(row[best_j].item())
            if best_sim >= merge_sim_thresh and best_j != cid and best_j not in merged:
                decisions.append({
                    "cluster_id": cid, "decision": "merge", "target_cluster_id": best_j,
                    "summary": f"geometric evidence: centroid cosine {best_sim:.3f} >= {merge_sim_thresh}.",
                    "confidence": "weak",
                })
                merged.add(cid)
                merged.add(best_j)
            elif variances[cid] >= split_thresh and variances[cid] > 0:
                decisions.append({
                    "cluster_id": cid, "decision": "split", "target_cluster_id": None,
                    "summary": f"geometric evidence: intra-cluster variance {variances[cid]:.4f} is high.",
                    "confidence": "weak",
                })
            else:
                decisions.append({
                    "cluster_id": cid, "decision": "keep", "target_cluster_id": None,
                    "summary": "geometric: cluster is compact and distinct.",
                    "confidence": "weak",
                })
        return decisions

    def get_decisions(self, packets, packet_dump_path=None, packet_fingerprint=None):
        """根据配置获取在线或离线决策。"""
        if not self.args.llm_enabled:
            return []
        if self.epoch == -1:
            return []

        if packet_fingerprint is None:
            packet_fingerprint = self.compute_packet_fingerprint(packets)

        if self.args.llm_mode == "online":
            try:
                decisions = self.fetch_online_decisions(packets, packet_fingerprint)
                if decisions:
                    return decisions
                print("Online LLM refinement returned no valid decisions.")
                return []
            except TimeoutError as exc:
                print(f"Online LLM refinement timed out: {exc}. Falling back to all-keep decisions.")
                return self._make_keep_decisions(packets)
            except Exception as exc:
                import socket as _socket
                if isinstance(exc, _socket.timeout):
                    print(f"Online LLM refinement timed out: {exc}. Falling back to all-keep decisions.")
                    return self._make_keep_decisions(packets)
                print(f"Online LLM refinement failed: {exc}")
                if self.args.llm_fallback:
                    decisions = self.load_decisions(expected_packet_fingerprint=packet_fingerprint)
                    if decisions:
                        print("Falling back to replayed JSON LLM decisions.")
                        return self.validate_decisions(decisions, max_clusters=len(packets))
                if self.args.llm_fail_open:
                    print(
                        "Fail-open enabled. Continuing without LLM decisions. "
                        f"Cluster packets are available at {packet_dump_path}."
                    )
                    return []
                raise

        decisions = self.load_decisions(expected_packet_fingerprint=packet_fingerprint)
        if not decisions:
            print(
                "LLM refinement enabled but no decisions file found. "
                f"Cluster packets dumped to {packet_dump_path}."
            )
            return []
        return self.validate_decisions(decisions, max_clusters=len(packets))

    def refine(self, features, examples, assignments, centroids):
        """执行未知簇的构包、决策与修正流程。

        由 args.refine_mode 选择四种重组依据：
          - none      : w/o Refine        —— 保持 KMeans 几何簇，不重组
          - random    : +Random Refine    —— 随机 keep/merge/split
          - geometric : +Geometric Refine —— 纯几何规则 merge/split
          - llm       : +LLM Refine (ours)—— LLM 话语功能级语义重组
        除 none 外，三种模式的成员重组后均经 apply_decisions 内部的
        数量归一统一回真实未知类数（strong 簇受保护，不被归一触碰）。
        """
        mode = getattr(self.args, "refine_mode", "none")

        # w/o Refine：KMeans 已产生 = 真实未知类数 的簇，直接沿用，不重组、不归一。
        if mode == "none":
            print("[Refine] mode=none (w/o Refine): keep KMeans clusters unchanged.")
            packets = self.packet_builder.build(features, examples, assignments, centroids)
            summaries = [None] * centroids.size(0)
            return assignments, centroids, packets, summaries

        packets = self.packet_builder.build(features, examples, assignments, centroids)
        packets = self.attach_training_progress_to_packets(packets)

        if mode == "random":
            print("[Refine] mode=random (+Random Refine).")
            decisions = self._random_decisions(packets)
        elif mode == "geometric":
            print("[Refine] mode=geometric (+Geometric Refine).")
            decisions = self._geometric_decisions(features, assignments, centroids)
        elif mode == "llm":
            print("[Refine] mode=llm (+LLM Refine, ours).")
            packet_dump_path, packet_fingerprint, decision_packets = self.dump_packets(packets)
            if packet_fingerprint is None:
                packet_fingerprint = self.compute_packet_fingerprint(packets)
                decision_packets = packets
            decisions = self.get_decisions(
                decision_packets,
                packet_dump_path=packet_dump_path,
                packet_fingerprint=packet_fingerprint,
            )
        else:
            raise ValueError(f"Unknown refine_mode: {mode}")

        refined_assignments, refined_centroids, refined_lineage = self.apply_decisions(
            features, assignments, centroids, decisions
        )

        refined_packets = self.packet_builder.build(
            features,
            examples,
            refined_assignments,
            refined_centroids,
        )
        refined_packets = self.attach_training_progress_to_packets(refined_packets)

        summaries = self.collect_summaries(refined_lineage, decisions)
        return refined_assignments, refined_centroids, refined_packets, summaries

    def collect_summaries(self, lineage, decisions):
        """汇总修正后各簇的文本摘要。"""
        summaries = [None for _ in range(len(lineage))]
        for final_cluster_id, source_clusters in enumerate(lineage):
            for decision in decisions:
                cluster_id = decision.get("cluster_id")
                target_cluster_id = decision.get("target_cluster_id")
                if cluster_id in source_clusters or target_cluster_id in source_clusters:
                    summary = decision.get("summary")
                    if summary:
                        summaries[final_cluster_id] = summary
                        break
        return summaries

    @staticmethod
    def _build_orig_confidence(decisions):
        """由决策列表构建 原始簇id -> 置信度等级(int) 映射。

        未出现在决策中的簇默认 0（fallback，最该被数量归一调整）。
        同一簇多条决策时取最具保护性的等级。
        """
        orig_confidence = {}
        for d in decisions:
            cid = d.get("cluster_id")
            if not isinstance(cid, int):
                continue
            lvl = _CONF_RANK.get(str(d.get("confidence", "weak")), 1)
            orig_confidence[cid] = max(orig_confidence.get(cid, 0), lvl)
        return orig_confidence

    def apply_decisions(self, features, assignments, centroids, decisions):
        # 决策置信度：原始簇id -> 等级。strong(2) 簇在数量归一时锁定不动。
        orig_confidence = self._build_orig_confidence(decisions)

        current_assignments = assignments.clone()
        current_centroids = centroids.clone()
        current_lineage = [[cluster_id] for cluster_id in range(centroids.size(0))]

        merge_pairs = []
        split_original_ids = []   # 记录原始 cluster_id，用于 merge 后重映射
        for decision in decisions:
            decision_type = decision.get("decision")
            cluster_id = decision.get("cluster_id")
            target_cluster_id = decision.get("target_cluster_id")

            if decision_type == "merge":
                if isinstance(cluster_id, int) and isinstance(target_cluster_id, int):
                    merge_pairs.append((cluster_id, target_cluster_id))
            elif decision_type == "split":
                if isinstance(cluster_id, int):
                    split_original_ids.append(cluster_id)

        if merge_pairs:
            current_assignments, current_centroids, current_lineage = self.apply_merges(
                features, current_assignments, current_centroids, merge_pairs, current_lineage
            )

        # merge 后将原始 split cluster_id 映射到新 id：
        # current_lineage[new_id] 包含了该新簇所合并的所有原始 cluster_id
        split_requests = []
        for orig_id in split_original_ids:
            for new_id, sources in enumerate(current_lineage):
                if orig_id in sources:
                    if new_id not in split_requests:
                        split_requests.append(new_id)
                    break

        if split_requests:
            current_assignments, current_centroids, current_lineage = self.apply_splits(
                features, current_assignments, current_centroids, split_requests, current_lineage
            )

        # 只修复空簇等异常，不强制恢复到原始簇数
        current_assignments, current_centroids, current_lineage = self.fix_empty_clusters(
            features, current_assignments, current_centroids, current_lineage
        )
        # 消解样本数 < 10 的小簇：成员并入最近邻簇，小簇直接消失
        current_assignments, current_centroids, current_lineage = self.dissolve_small_clusters(
            features, current_assignments, current_centroids, current_lineage
        )

        # ── 解耦设计：LLM 主导语义重组后，数量归一只把基数对齐回固定的
        #    真实未知类数；strong 簇（三票一致 keep / split 强证据产物）锁定，
        #    归一只在 fallback/weak 残余簇上进行，不破坏语义分组。
        if self._true_unknown_count and self._true_unknown_count >= 1:
            current_assignments, current_centroids, current_lineage = self.normalize_cluster_count(
                features,
                current_assignments,
                current_centroids,
                self._true_unknown_count,
                current_lineage,
                orig_confidence=orig_confidence,
            )

        return current_assignments, current_centroids, current_lineage

    def apply_merges(self, features, assignments, centroids, merge_pairs, lineage):
        """根据合并对更新聚类分配与中心。"""
        parent = list(range(centroids.size(0)))

        def find(x):
            """查找并返回并查集根节点。"""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            """合并两个并查集节点。"""
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in merge_pairs:
            if a < len(parent) and b < len(parent):
                union(a, b)

        remapped = torch.tensor([find(int(idx)) for idx in assignments.cpu().tolist()], dtype=torch.long)
        unique_roots = sorted(set(remapped.tolist()))
        root_to_new = {root: i for i, root in enumerate(unique_roots)}
        new_assignments = torch.tensor([root_to_new[val] for val in remapped.tolist()], device=assignments.device)
        new_centroids = self.compute_centroids(features, new_assignments, len(unique_roots))
        new_lineage = []
        for root in unique_roots:
            members = []
            for idx in range(len(parent)):
                if find(idx) == root:
                    members.extend(lineage[idx])
            new_lineage.append(sorted(set(members)))
        return new_assignments, new_centroids, new_lineage

    def apply_splits(self, features, assignments, centroids, split_requests, lineage):
        """根据拆分请求细化目标簇。

        修复说明：
        1. 移除 target_clusters 参数，改为对每个请求独立尽力执行，
           不再因计算错误的 target 而提前 break。
        2. lineage 追加时分别记录两个子簇各自的来源，而非都继承父簇全部 lineage。
        """
        current_assignments = assignments.clone()
        current_num_clusters = centroids.size(0)
        current_lineage = [list(item) for item in lineage]

        for cluster_id in split_requests:
            member_mask = current_assignments == cluster_id
            member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
            if member_indices.numel() < 4:
                continue
            cluster_feats = features[member_indices].cpu().numpy()
            local_km = KMeans(n_clusters=2, n_init=10, random_state=self.args.seed)
            with threadpool_limits(limits=1):
                local_km.fit(cluster_feats)
            local_labels = local_km.labels_
            if local_labels is None:
                continue
            if len(np.unique(local_labels)) < 2:
                continue
            # 子簇 0：留在原 cluster_id；子簇 1：分配到新 id
            new_cluster_id = current_num_clusters
            for pos, member_idx in enumerate(member_indices.cpu().tolist()):
                if local_labels[pos] == 1:
                    current_assignments[member_idx] = new_cluster_id
            # 两个子簇各自独立记录 lineage，都来源于父簇
            parent_lineage = list(current_lineage[cluster_id])
            current_lineage[cluster_id] = parent_lineage   # 子簇 0 保留父 lineage
            current_lineage.append(parent_lineage)          # 子簇 1 同样来自父 lineage
            current_num_clusters += 1

        new_centroids = self.compute_centroids(features, current_assignments, current_num_clusters)
        return current_assignments, new_centroids, current_lineage

    def fix_empty_clusters(self, features, assignments, centroids, lineage):
        """修复聚类结果中的空簇：将空簇重新分配给最近样本。

        仅处理异常情况（空簇），不改变簇的总数量。
        这是 apply_decisions 流程的最终清理步骤，保证每个簇至少有一个样本。
        """
        current_assignments = assignments.clone()
        current_centroids = centroids.clone()
        current_lineage = [list(item) for item in lineage]
        num_clusters = current_centroids.size(0)

        for cluster_id in range(num_clusters):
            member_mask = current_assignments == cluster_id
            if member_mask.sum() == 0:
                # 空簇：将特征空间中最近的样本重新划入此簇
                dists = torch.cdist(
                    features.unsqueeze(0),
                    current_centroids[cluster_id].unsqueeze(0).unsqueeze(0)
                ).squeeze()
                nearest = torch.argmin(dists).item()
                current_assignments[nearest] = cluster_id

        # 重新计算质心（处理空簇后）
        current_centroids = self.compute_centroids(features, current_assignments, num_clusters)
        return current_assignments, current_centroids, current_lineage

    def dissolve_small_clusters(self, features, assignments, centroids, lineage, min_size=10):
        """消解样本数不足 min_size 的小簇。

        策略：
        1. 将小簇的每个成员样本按余弦相似度分配给最近的其他簇。
        2. 小簇本身从 id 空间中移除，剩余簇重新连续编号。
        3. 迭代执行直到所有簇均满足 min_size（防止消解后再出现新的小簇）。
        """
        current_assignments = assignments.clone()
        current_lineage = [list(item) for item in lineage]

        while True:
            num_clusters = int(current_assignments.max().item()) + 1
            sizes = torch.zeros(num_clusters, dtype=torch.long)
            for cid in range(num_clusters):
                sizes[cid] = (current_assignments == cid).sum()

            small_ids = [cid for cid in range(num_clusters) if sizes[cid] < min_size]
            if not small_ids:
                break

            # 将所有有效簇（非小簇）的质心重新计算作为目标
            valid_ids = [cid for cid in range(num_clusters) if cid not in small_ids]
            if not valid_ids:
                # 极端情况：所有簇都是小簇，无法消解，直接退出
                break

            valid_centroids = self.compute_centroids(features, current_assignments, num_clusters)[valid_ids]
            norm_valid_centroids = torch.nn.functional.normalize(valid_centroids, dim=1)
            norm_features = torch.nn.functional.normalize(features, dim=1)

            for small_id in small_ids:
                member_mask = current_assignments == small_id
                member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
                if member_indices.numel() == 0:
                    continue
                # 每个成员找余弦最近的有效簇
                member_feats = norm_features[member_indices]          # (M, D)
                sims = torch.matmul(member_feats, norm_valid_centroids.T)  # (M, V)
                nearest_valid_pos = torch.argmax(sims, dim=1)        # (M,)
                for i, member_idx in enumerate(member_indices.cpu().tolist()):
                    current_assignments[member_idx] = valid_ids[nearest_valid_pos[i].item()].item()

            # 压缩 id：只保留 valid_ids，重新映射为连续 0..len(valid_ids)-1
            old_to_new = {old: new for new, old in enumerate(valid_ids)}
            new_assignments_list = [old_to_new[int(a)] for a in current_assignments.cpu().tolist()]
            current_assignments = torch.tensor(new_assignments_list, dtype=torch.long, device=assignments.device)
            current_lineage = [current_lineage[old] for old in valid_ids]

        current_centroids = self.compute_centroids(features, current_assignments, int(current_assignments.max().item()) + 1)
        return current_assignments, current_centroids, current_lineage

    def normalize_cluster_count(self, features, assignments, centroids,
                                target_clusters, lineage, orig_confidence=None):
        """将簇数量归一到目标数量。

        解耦设计：归一只把“基数”对齐回 target_clusters（=真实未知类数），
        并尽量不破坏 LLM 已确立的语义分组——
          - strong 簇（三票一致 keep / split 强证据产物）：锁定，归一绝不触碰；
          - 残余簇优先级（最该被调整 -> 最该保护）：
              fallback(三票全异回退 keep / 未审查) > weak(两票共识) > strong(锁定)。
        orig_confidence 为 None 时退化为不区分等级的纯几何归一（向后兼容）。
        """
        current_assignments = assignments.clone()
        current_centroids = centroids.clone()
        current_lineage = [list(item) for item in lineage]

        def level_of(lineage_entry):
            if not orig_confidence:
                return 0
            return max((orig_confidence.get(int(s), 0) for s in lineage_entry), default=0)

        # ── 簇数不足：split 残余簇（优先 fallback，其次 weak，绝不动 strong）──
        while current_centroids.size(0) < target_clusters:
            n = current_centroids.size(0)
            cand = []
            for cid in range(n):
                if level_of(current_lineage[cid]) >= 2:
                    continue  # strong：锁定
                member_mask = current_assignments == cid
                member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
                if member_indices.numel() < 4:
                    continue
                feats = features[member_indices]
                var = torch.mean((feats - current_centroids[cid].unsqueeze(0)).pow(2)).item()
                cand.append((cid, level_of(current_lineage[cid]), var))
            if not cand:
                break
            # 先按置信度升序（fallback 优先），同级再按方差降序
            cand.sort(key=lambda x: (x[1], -x[2]))
            split_id, _, split_var = cand[0]
            if split_var <= 0:
                break
            current_assignments, current_centroids, current_lineage = self.apply_splits(
                features, current_assignments, current_centroids, [split_id], current_lineage
            )

        # ── 簇数过多：merge 残余簇（优先 fallback 之间，绝不动 strong）──
        while current_centroids.size(0) > target_clusters:
            n = current_centroids.size(0)
            non_strong = [cid for cid in range(n) if level_of(current_lineage[cid]) < 2]
            tier0 = [cid for cid in non_strong if level_of(current_lineage[cid]) == 0]
            if len(tier0) >= 2:
                pool = tier0                       # 优先在 fallback 簇之间合并
            elif len(non_strong) >= 2:
                pool = non_strong                  # 其次在 fallback+weak 之间合并
            else:
                pool = list(range(n))              # 兜底：必须满足基数，允许触碰 strong

            norm_c = F.normalize(current_centroids, dim=1)
            sim = torch.matmul(norm_c, norm_c.T)
            sim.fill_diagonal_(-1e9)

            best_pair = None
            best_sim = -1e18
            for ai in range(len(pool)):
                for bi in range(ai + 1, len(pool)):
                    a, b = pool[ai], pool[bi]
                    s = float(sim[a, b].item())
                    if s > best_sim:
                        best_sim = s
                        best_pair = (a, b)
            if best_pair is None:
                break
            current_assignments, current_centroids, current_lineage = self.apply_merges(
                features, current_assignments, current_centroids, [best_pair], current_lineage
            )

        return current_assignments, current_centroids, current_lineage

    @staticmethod
    def compute_centroids(features, assignments, num_clusters):
        """根据样本分配结果重新计算簇中心。"""
        centroids = []
        for cluster_id in range(num_clusters):
            member_mask = assignments == cluster_id
            member_indices = torch.nonzero(member_mask, as_tuple=False).squeeze(-1)
            if member_indices.numel() == 0:
                centroids.append(torch.zeros(features.size(1), device=features.device))
                continue
            center = features[member_indices].mean(dim=0)
            centroids.append(F.normalize(center, dim=0))
        return torch.stack(centroids, dim=0)
