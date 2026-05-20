"""训练流水线编排。

该模块负责串联三阶段训练流程，并处理分布式初始化、缓存加载与最终评估。
"""

import os

import torch

from data_modules import Data
from config.arguments import init_model
from managers.external_pretrain_manager import ExternalPretrainModelManager
from managers.internal_pretrain_manager import InternalPretrainModelManager
from managers.noise_manager import NoiseManager
from utils.build_ml import set_args
from utils.output_paths import OutputPaths
from utils.parallel import barrier, cleanup_distributed, get_model_backbone, get_wrapped_state_dict, init_distributed_mode, is_main_process, load_wrapped_model, save_wrapped_backbone, save_wrapped_model, unwrap_parallel_model


def _run_stage1_pretraining(args, data):
    """执行阶段一的外部与内部预训练流程。"""
    output_paths = OutputPaths(args)
    external_pretrain = output_paths.external_pretrain_cache_path()
    manager_ext = ExternalPretrainModelManager(
        args,
        data.external_train_dataloader,
        data.external_eval_dataloader,
        data.external_num_labels
    )
    if manager_ext.model is None:
        raise ValueError("External pretrain model is not initialized.")

    if os.path.exists(external_pretrain):
        print(f"\nFound saved external pretrain cache. Loading from: {external_pretrain}")
        load_wrapped_model(manager_ext.model, external_pretrain)
        print("Step 1 (External) skipped — loaded from cache.")
    else:
        print('\n[Step 1] External Pre-training begin...')
        manager_ext.train(args)
        if is_main_process():
            output_paths.ensure_parent_dir(external_pretrain)
            save_wrapped_model(unwrap_parallel_model(manager_ext.model), external_pretrain)
        barrier()
        # 所有进程重新从磁盘加载，确保权重完全一致。
        load_wrapped_model(manager_ext.model, external_pretrain)
        print('[Step 1] External Pre-training finished.')

    pretrained_model_from_stage1 = manager_ext.model
    if pretrained_model_from_stage1 is None:
        raise ValueError("Stage 1 pretrained model is not available.")

    if args.known_cls_ratio == 0:
        args.disable_pretrain = True
        print("[Stage 1] Internal Pre-training SKIPPED (known_cls_ratio is 0).")
        return pretrained_model_from_stage1

    args.disable_pretrain = False
    print('\n[Step 1] Internal Pre-training begin...')
    manager_p = InternalPretrainModelManager(args, data)
    if manager_p.model is None:
        raise ValueError("Internal pretrain model is not initialized.")

    if args.step2_ckpt and os.path.exists(args.step2_ckpt):
        print(f"Loading cached internal pretrain model from: {args.step2_ckpt}")
        load_wrapped_model(manager_p.model, args.step2_ckpt)
    else:
        # 用外部预训练的 backbone 权重初始化内部预训练模型。
        pretrained_dict = get_model_backbone(pretrained_model_from_stage1).state_dict()
        get_model_backbone(manager_p.model).load_state_dict(pretrained_dict, strict=True)
        manager_p.train(args, data)

        if args.step2_ckpt:
            print(f"Saving internal pretrain model to: {args.step2_ckpt}")
            if is_main_process():
                output_paths.ensure_parent_dir(args.step2_ckpt)
                save_wrapped_model(unwrap_parallel_model(manager_p.model), args.step2_ckpt)
            barrier()
            # 所有进程重新从磁盘加载，确保权重完全一致。
            load_wrapped_model(manager_p.model, args.step2_ckpt)

    print('Internal Pre-training finished!')
    return manager_p.model


def _run_stage2_distillation(args, data, final_pretrained_model):
    """执行阶段二教师到学生的蒸馏流程。"""
    if final_pretrained_model is None:
        raise ValueError("Final pretrained model is required for stage 2.")

    output_paths = OutputPaths(args)
    manager_d = NoiseManager(args, data)
    if args.view_strategy == "SADA":
        print('\nLoading Teacher model backbone...')
        pretrained_final = get_model_backbone(final_pretrained_model).state_dict()
        get_model_backbone(manager_d.teacher_model).load_state_dict(pretrained_final, strict=True)

        final_model_dict = get_wrapped_state_dict(final_pretrained_model)

        if 'classifier.weight' in final_model_dict:
            print("Loading classifier...")
            ckpt_weight = final_model_dict['classifier.weight']
            ckpt_bias = final_model_dict.get('classifier.bias', None)
            model_classifier = unwrap_parallel_model(manager_d.teacher_model).classifier

            # 教师模型分类头维度 == n_known_cls，Stage1 checkpoint 也是 n_known_cls 行，
            # 形状完全匹配，直接整体加载即可，无需按标签名逐行对齐。
            if model_classifier.weight.shape == ckpt_weight.shape:
                with torch.no_grad():
                    model_classifier.weight.copy_(ckpt_weight)
                    if ckpt_bias is not None and model_classifier.bias is not None:
                        model_classifier.bias.copy_(ckpt_bias)
                print(f"Teacher Classifier Head: all {ckpt_weight.shape[0]} known classes loaded successfully.")
            else:
                # 形状不符时退回到逐标签对齐（兜底）
                print(f"[Warning] Classifier shape mismatch "
                      f"(model={model_classifier.weight.shape}, ckpt={ckpt_weight.shape}), "
                      f"falling back to label-wise alignment.")
                src_label_list = data.known_label_list
                known_label2id = {label: i for i, label in enumerate(src_label_list)}
                loaded_count = 0
                with torch.no_grad():
                    for src_idx, label_name in enumerate(src_label_list):
                        if src_idx < ckpt_weight.shape[0] and src_idx < model_classifier.weight.shape[0]:
                            if model_classifier.weight.shape[1] == ckpt_weight.shape[1]:
                                model_classifier.weight[src_idx] = ckpt_weight[src_idx]
                                if ckpt_bias is not None and model_classifier.bias is not None:
                                    model_classifier.bias[src_idx] = ckpt_bias[src_idx]
                                loaded_count += 1
                print(f"Teacher Classifier Head: {loaded_count}/{len(src_label_list)} known classes loaded.")
        else:
            print("[Warning] No classifier found in checkpoint. Using random initialization for Head.")

        if args.step3_ckpt and os.path.exists(args.step3_ckpt):
            print(f"\n[Step 2] Found cached Student Model at: {args.step3_ckpt}")
            state_dict = torch.load(args.step3_ckpt, map_location="cpu")
            unwrap_parallel_model(manager_d.student_model).load_state_dict(state_dict)
        else:
            print('\n[Step 2] Distillation begin...')
            manager_d.Mask_BERT_with_ratio(args, data)
            print('\n[Step 2] Distillation finish...')

            if args.step3_ckpt:
                print(f"Saving Step 3 Student Model to: {args.step3_ckpt}")
                if is_main_process():
                    output_paths.ensure_parent_dir(args.step3_ckpt)
                    torch.save(get_wrapped_state_dict(manager_d.student_model), args.step3_ckpt)
                barrier()
                # 所有进程统一从磁盘加载，确保多卡权重一致且 map_location 安全。
                state_dict = torch.load(args.step3_ckpt, map_location="cpu")
                unwrap_parallel_model(manager_d.student_model).load_state_dict(state_dict)
    else:
        print(f"Strategy is '{args.view_strategy}', skipping Step 2.")

    if args.view_strategy == "SADA":
        return manager_d.student_model
    return None


def _run_stage3_training_eval(args, data, final_pretrained_model, student_model, sada_manager_cls):
    """执行阶段三训练、保存与评估流程。"""
    manager_sada = sada_manager_cls(args, data, final_pretrained_model, student_model=student_model)

    if args.report_pretrain:
        original_method = args.method
        args.method = 'pretrain_eval'
        print('Evaluating model performance before training...')
        try:
            manager_sada.evaluation(args, data)
        finally:
            args.method = original_method
        barrier()

    print('\n[Step 3] SADA Training begin...')
    proto_manager = manager_sada.train(args, data)
    print('[Step 3] Finished.')

    # ── 保存训练过程中的最佳评估结果到 CSV ────────────────────────────────
    if manager_sada.best_eval_results is not None:
        manager_sada.test_results = manager_sada.best_eval_results
        manager_sada.save_results(args)
        print(f"\n[Best eval results saved] {manager_sada.best_eval_results}")

    # ── 最终评估（仅打印输出，不重复写 CSV）─────────────────────────────
    print('\n Final Evaluation begin...')
    manager_sada.evaluation(args, data, save_results=False, proto_manager=proto_manager)
    barrier()

    # ── 保存最终模型 ──────────────────────────────────────────────────────────
    print('\n Saving Final Model...')
    try:
        if is_main_process():
            save_wrapped_backbone(manager_sada.model, args.save_model_path)
        barrier()
        print(f"\n Final model saved to {args.save_model_path}")
    except Exception as e:
        print(f"\n[Warning] Model save failed: {e}. Evaluation results are already logged above.")

    print("\n Visualizing learned features...")
    manager_sada.visualize_Tsne(data=data, args=args)
    barrier()


def run_pipeline(sada_manager_cls):
    """串联整个项目的训练与评估流水线。"""
    parser = init_model()
    args = parser.parse_args()

    # 仅在非 torchrun 场景设置 CUDA_VISIBLE_DEVICES。
    # DDP（torchrun）会通过 LOCAL_RANK/WORLD_SIZE 管理设备映射；
    # 子进程内再次改写 CUDA_VISIBLE_DEVICES 可能导致 local_rank 映射异常，
    # 进而引发多进程初始化阶段崩溃（含 SIGSEGV）。
    world_size_env = os.environ.get("WORLD_SIZE")
    is_torchrun = world_size_env is not None and world_size_env != "1"
    if getattr(args, "gpu_ids", None) and not is_torchrun:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    init_distributed_mode(args)

    try:
        print('[Init] Data and Parameters Initialization...')
        if getattr(args, "distributed", False):
            args.device = torch.device(f"cuda:{args.local_rank}")
        else:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_args(args)
        output_paths = OutputPaths(args)

        if is_main_process():
            output_paths.model_dir()
        barrier()

        data = Data(args)
        args.label2id = {label: i for i, label in enumerate(data.all_label_list)}
        args.id2label = {i: label for i, label in enumerate(data.all_label_list)}

        print(args)

        final_pretrained_model = _run_stage1_pretraining(args, data)
        student_model = _run_stage2_distillation(args, data, final_pretrained_model)
        _run_stage3_training_eval(args, data, final_pretrained_model, student_model, sada_manager_cls)

        print("\n All Finished!")
        # numba/LLVM JIT 的析构在 torchrun 多进程退出时会触发 SIGSEGV（numba#3749）。
        # cleanup_distributed 完成后直接 os._exit(0) 跳过 Python GC 析构链，
        # 确保所有分布式 collective 已完成但不触发 native 析构 crash。
        cleanup_distributed()
        os._exit(0)
    finally:
        # 仅在 try 块内抛异常时执行（正常路径已由上方 os._exit 终止）
        cleanup_distributed()
