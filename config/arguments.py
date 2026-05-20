from argparse import ArgumentParser


def init_model():
    """构建项目统一命令行参数解析器。"""
    parser = ArgumentParser()
    
    parser.add_argument("--internal_data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--internal_dataset", default=None, type=str, required=True,
                        help="Name of dataset.")

    parser.add_argument("--internal_max_seq_length", default=30, type=int,
                        help="Maximum sequence length for the INTERNAL dataset.")

    parser.add_argument("--step2_ckpt", type=str, default=None,
                        help="Path to cache/load Step 2 internal pretrained model.")

    parser.add_argument("--step3_ckpt", type=str, default=None,
                        help="Path to cache/load the distilled student noise model used by Stage 3 dual-view augmentation.")
    
    parser.add_argument("--save_results_path", type=str, default='outputs',
                        help="The path to save results.")

    parser.add_argument("--external_data_dir", default='data_external', type=str,
                        help="The root input data dir for the EXTERNAL pre-training task.")

    parser.add_argument("--external_dataset", default='clinc', type=str,
                        help="Name of EXTERNAL dataset for pre-training (e.g., clinc).")

    parser.add_argument("--external_max_seq_length", default=30, type=int,
                        help="Maximum sequence length for the EXTERNAL dataset.")

    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="The path or name for the pre-trained bert model.")

    parser.add_argument("--tokenizer", default="bert-base-uncased", type=str,
                        help="The path or name for the tokenizer")
    
    parser.add_argument("--feat_dim", default=256, type=int,
                        help="Contrastive learning projection head output dimension.")
    
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Warmup proportion for optimizer.")

    parser.add_argument("--save_model_path", default='./save_models/final/', type=str,
                        help="Path to save model checkpoints. Set to None if not save.")
    
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, required=True,
                        help="The ratio of known classes.")

    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed.")

    parser.add_argument("--gpu_ids", default=None, type=str,
                        help="Comma-separated CUDA device IDs, e.g. 0,1,3. Overrides CUDA_VISIBLE_DEVICES if set before CUDA initialization.")

    parser.add_argument("--local_rank", default=0, type=int,
                        help="Local rank for torchrun/DDP.")

    parser.add_argument("--distributed", action="store_true",
                        help="Enable DistributedDataParallel when launched with torchrun.")

    parser.add_argument("--method", type=str, default='SADA',
                        help="The name of method.")

    parser.add_argument("--labeled_ratio", default=0.1, type=float,
                        help="The ratio of labeled samples.")
    
    parser.add_argument("--rtr_prob", default=0.25, type=float,
                        help="Probability for random token replacement")

    parser.add_argument("--pretrain_batch_size", default=64, type=int,
                        help="Batch size for pre-training")

    parser.add_argument("--distillation_batch_size", default=128, type=int,
                        help="Batch size for distillation")

    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")

    parser.add_argument("--wait_patient", default=20, type=int,
                        help="Patient steps for Early Stop in pretraining.") 

    parser.add_argument("--num_pretrain_epochs", default=100, type=int,
                        help="The pre-training epochs.")

    parser.add_argument("--num_distillate_epochs", default=50, type=int,
                        help="The distillation epochs.")

    parser.add_argument("--num_train_epochs", default=34, type=int,
                        help="The training epochs.")

    parser.add_argument("--lr_pre", default=5e-5, type=float,
                        help="The learning rate for pre-training.")

    parser.add_argument("--lr_distill", default=1e-3, type=float,
                        help="The learning rate for Stage 2 student (NoiseGenerator) distillation. "
                             "NoiseGenerator is randomly initialized, so it needs a much larger lr than fine-tuning.")

    parser.add_argument("--lr", default=1e-5, type=float,
                        help="The learning rate for training.")
        
    parser.add_argument("--temp", default=0.07, type=float,
                        help="Temperature for contrastive and prototype assignment losses.")

    parser.add_argument("--view_strategy", default="SADA", type=str,
                        help="Dual-view construction strategy for Stage 3. Choose from SADA|rtr|shuffle|none.")

    parser.add_argument("--input_strategy", default="CONTEXT", type=str,
                        help="input_strategy")

    parser.add_argument("--with_speaker", default=1, type=int,
                        help="with_speaker (1 for True, 0 for False)")

    parser.add_argument("--report_pretrain", action="store_true",
                        help="Enable reporting performance right after pretrain")
    
    parser.add_argument("--disable_pretrain", action="store_true",
                                    help="Skip Step 1 (Pre-training)")

    parser.add_argument("--grad_clip", default=1, type=float,
                        help="Value for gradient clipping.")

    parser.add_argument("--noise_ratio", "--ratio", "--mask_threshold", dest="noise_ratio", default=0.75, type=float,
                        help="Unified noise ratio used consistently in teacher noise labeling, student noise modeling, and Stage 3 whole-word masking.")

    parser.add_argument("--alpha", default=1.0, type=float,
                        help="Weight for distillation loss.")

    parser.add_argument("--with_pos", action="store_true",
                        help="Enable POS tagging filter for IG mask.")

    parser.add_argument("--with_abs", action="store_true",
                        help="Use absolute value for IG scores.")

    parser.add_argument("--is_continuous", action="store_true", default=False,
                        help="Force continuous masking (span masking).")

    parser.add_argument("--with_mask", action="store_true",
                        help="Enable masking for visualization/inference.")

    parser.add_argument("--use_transformer_student", action="store_true", default=True,
                    help="Use Transformer encoder in student model instead of MLP")
    
    parser.add_argument("--student_num_heads", default=4, type=int,
                        help="Number of attention heads in student transformer")
    
    parser.add_argument("--student_num_layers", default=2, type=int,
                        help="Number of transformer layers in student model")
    
    parser.add_argument("--confidence_gate_threshold", default=0.2, type=float,
                        help="Confidence threshold for gated teacher-to-student mask distillation.")
    
    parser.add_argument("--loss_cl_weight", default=0.5, type=float,
                        help="Weight for the main contrastive loss (loss_cl) in Stage 3. Fixed throughout training.")

    parser.add_argument("--proto_loss_weight", default=0.5, type=float,
                    help="Weight for labeled prototype contrastive loss in Stage 3.")
    
    parser.add_argument("--proto_momentum", default=0.9, type=float,
                        help="Momentum for known and unknown prototype updates.")
    
    parser.add_argument("--use_proto_init_kmeans", action="store_true", 
                        default=True,
                        help="Initialize unknown-cluster refresh and final KMeans with current prototype vectors when available.")

    parser.add_argument("--refine_start_epoch", default=10, type=int,
                        help="Epoch index at which periodic unknown-cluster refresh begins in Stage 3.")

    parser.add_argument("--refine_every", default=10, type=int,
                        help="Refresh unknown clusters and unknown prototypes every N epochs after refinement starts.")

    parser.add_argument("--packet_topk_representatives", default=8, type=int,
                        help="Number of representative examples included per unknown-cluster packet.")

    # ================== 簇细化模式（四组对照实验开关）==================
    # none      : w/o Refine          —— 不重组，保持 KMeans 几何簇
    # random    : +Random Refine      —— 随机 keep/merge/split（排除“任何重组都有效”）
    # geometric : +Geometric Refine   —— 簇内方差/簇间相似度的纯几何规则重组
    # llm       : +LLM Refine (ours)  —— LLM 话语功能级语义重组
    # 四组最终簇数均归一到真实未知类数，唯一变量是“成员重组依据”。
    parser.add_argument("--refine_mode", default="none", type=str,
                        choices=["none", "random", "geometric", "llm"],
                        help="Stage-3 unknown-cluster refinement mode: "
                             "none=w/o Refine, random=+Random Refine, "
                             "geometric=+Geometric Refine, llm=+LLM Refine (ours).")

    parser.add_argument("--llm_enabled", action="store_true",
                        help="Enable cluster-level LLM refinement. Use with --llm_mode to choose online API or offline replay.")

    parser.add_argument("--llm_mode", default="offline", type=str,
                        choices=["offline", "online"],
                        help="LLM refinement mode: offline JSON replay or online HTTP API call.")

    parser.add_argument("--llm_decisions", default="outputs/llm_cluster_decisions.json", type=str,
                        help="Path to a JSON file containing replayable keep/merge/split decisions for unknown clusters.")

    parser.add_argument("--llm_url", default=None, type=str,
                        help="HTTP endpoint used for online cluster-level LLM refinement.")

    parser.add_argument("--llm_model", default=None, type=str,
                        help="Model name sent to the online LLM API payload.")

    parser.add_argument("--llm_key", default=None, type=str,
                        help="Bearer token used for online LLM API authentication. Prefer --llm_key_env when possible.")

    parser.add_argument("--llm_key_env", default="LLM_API_KEY", type=str,
                        help="Environment variable name used to resolve the online LLM API bearer token.")

    parser.add_argument("--llm_url_env", default="LLM_API_URL", type=str,
                        help="Environment variable name used to resolve the online LLM API URL.")

    parser.add_argument("--llm_model_env", default="LLM_API_MODEL", type=str,
                        help="Environment variable name used to resolve the online LLM API model name.")

    parser.add_argument("--llm_env_file", default=".env", type=str,
                        help="Path to a .env file used to load online LLM API configuration.")

    parser.add_argument("--llm_timeout", default=240, type=int,
                        help="Timeout in seconds for online LLM API requests.")

    parser.add_argument("--llm_headers", default=None, type=str,
                        help="Optional JSON object string of extra HTTP headers for the online LLM API.")

    parser.add_argument("--llm_fallback", action="store_true",
                        help="If online API refinement fails, fall back to --llm_decisions when available.")

    parser.add_argument("--llm_fail_open", action="store_true",
                        help="If online refinement fails and no replay JSON is available, continue training without LLM decisions.")

    parser.add_argument("--llm_only_suspicious", action="store_true",
                        help="Dump only suspicious unknown clusters for offline refinement review.")

    parser.add_argument("--llm_var_thresh", default=0.12, type=float,
                        help="Variance threshold used to mark a cluster as suspicious.")

    # ================== 新增：双分支路由与伪标签权重 ==================
    parser.add_argument("--compete_bias", type=float, default=0.10, help="未知类路由竞争时的偏置补偿")
    parser.add_argument("--known_sim_thresh", type=float, default=0.40, help="已知类相似度门控")
    parser.add_argument("--known_conf_thresh", type=float, default=0.15, help="已知类置信度(Gap)门控")
    parser.add_argument("--unknown_sim_thresh", type=float, default=0.20, help="未知类相似度门控")
    parser.add_argument("--unknown_conf_thresh", type=float, default=0.05, help="未知类置信度(跨空间Gap)门控")

    parser.add_argument("--pseudo_weight_known", type=float, default=1.0, help="已知类伪标签Loss权重")
    parser.add_argument("--pseudo_weight_unknown", type=float, default=0.8, help="未知类伪标签Loss权重")
    parser.add_argument("--speaker_mask_prob", type=float, default=0.3, help="对 speaker token（[s]/[t]）执行随机遮蔽的概率")
    parser.add_argument("--warmup_ucl_weight", type=float, default=0.05,
                        help="warmup 期对 unlabeled 样本施加同实例双视图 SimCLR 的权重，"
                             "给 unknown 特征提供最小梯度信号，防止其在 warmup 期漂离特征空间。"
                             "设为 0 等价于禁用（恢复旧行为）。建议范围 0.02~0.10。")
    
    parser.add_argument("--balance_penalty", type=float, default=0.3, help="反塌缩频率惩罚权重(越大越强制均衡)")
    parser.add_argument("--margin_scale", type=float, default=2)
        
    return parser
