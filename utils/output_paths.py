"""统一管理项目输出路径与命名规则。"""

import os


class OutputPaths:
    """集中构造模型、结果、缓存与可视化产物的路径。"""

    def __init__(self, args):
        """根据命令行参数初始化输出路径辅助器。"""
        self.args = args

    @staticmethod
    def ensure_dir(path):
        """确保目录存在并返回原路径。"""
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def ensure_parent_dir(file_path):
        """确保文件所在父目录存在并返回原路径。"""
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return file_path

    @staticmethod
    def external_pretrain_cache_path():
        """返回阶段一外部预训练缓存路径。"""
        return os.path.join("./save_models", "external_pretrain.pth")

    def results_dir(self):
        """返回结果输出目录。"""
        return self.ensure_dir(self.args.save_results_path)

    def model_dir(self):
        """返回模型输出目录。"""
        return self.ensure_dir(self.args.save_model_path)

    def mask_cache_path(self, noise_ratio, with_pos):
        """返回教师掩码缓存文件路径。"""
        file_name = (
            f"mask_cache_{self.args.internal_dataset}_"
            f"l{self.args.labeled_ratio}_k{self.args.known_cls_ratio}_"
            f"inp{self.args.input_strategy}_spk{int(self.args.with_speaker)}_"
            f"r{noise_ratio}_pos{with_pos}_seed{self.args.seed}.pt"
        )
        return os.path.join(self.results_dir(), file_name)

    def attribution_viz_dir(self):
        """返回教师归因 HTML 可视化目录。"""
        return self.ensure_dir(os.path.join(self.model_dir(), "attribution_viz"))

    def attribution_viz_file(self, viz_index):
        """返回单个教师归因 HTML 文件路径。"""
        return os.path.join(self.attribution_viz_dir(), f"viz_sample_{viz_index}.html")

    def stage2_ckpt_path(self):
        """返回阶段二缓存路径。"""
        return self.args.step2_ckpt

    def stage3_ckpt_path(self):
        """返回阶段三学生模型缓存路径。"""
        return self.args.step3_ckpt

    def results_csv_path(self):
        """返回实验结果 CSV 路径。"""
        return os.path.join(self.results_dir(), "results.csv")

    def umap_png_path(self, epoch=None, space="cls"):
        """返回 UMAP 可视化图片路径。

        Args:
            epoch: 训练轮次，不为 None 时在文件名中标明 epoch 编号。
            space: 特征空间标识，'cls' 表示 BERT CLS embedding，'proj' 表示 projection head 输出。
        """
        epoch_tag = f"_epoch{epoch}" if epoch is not None else ""
        file_name = (
            f"vis_{self.args.internal_dataset}_K{self.args.known_cls_ratio}_L{self.args.labeled_ratio}_"
            f"{self.args.view_strategy}_{self.args.input_strategy}_SP{self.args.with_speaker}_"
            f"TN{self.args.noise_ratio}_SN{self.args.noise_ratio}_S{self.args.seed}"
            f"_{space}{epoch_tag}.png"
        )
        return os.path.join(self.results_dir(), file_name)

    def intent_profiles_json_path(self):
        """返回意图画像 JSON 路径。"""
        file_name = (
            f"intent_profiles_{self.args.internal_dataset}_"
            f"k{self.args.known_cls_ratio}_l{self.args.labeled_ratio}.json"
        )
        return os.path.join(self.results_dir(), file_name)

    def cluster_packets_path(self, epoch=None):
        """返回簇信息包 JSON 路径。"""
        if epoch is not None:
            file_name = f"cluster_packets_epoch{epoch}.json"
        else:
            file_name = "cluster_packets.json"
        return os.path.join(self.results_dir(), file_name)

    def llm_decisions_write_path(self, epoch=None):
        """返回在线/离线决策写入路径。"""
        base_path = self.args.llm_decisions
        if epoch is not None and base_path:
            # 在文件名中插入 epoch 编号
            dir_name = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            name, ext = os.path.splitext(base_name)
            base_path = os.path.join(dir_name, f"{name}_epoch{epoch}{ext}")
        return base_path
