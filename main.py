"""
项目主入口
"""
import os

# 限制 BLAS/OMP 线程数，防止 OpenBLAS 预编译线程上限溢出导致 SIGSEGV。
# 必须在任何 numpy/sklearn/torch 导入之前强制覆盖（不用 setdefault，
# 因为 torchrun 子进程可能从父进程继承了更高的值）。
# OMP_NUM_THREADS=1 是最保险的设置：OpenBLAS 预编译上限通常为 64，
# 但多进程场景下每个进程的 OMP 线程数叠加很容易超限。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# 补齐其余常见 BLAS/OpenMP 运行时开关，避免子进程继承/重置差异
# 导致 OpenBLAS 线程上限告警并触发 native SIGSEGV。
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# generate_intent_profiles() / evaluate_cluster_stability()
from managers.sada_manager import SADAModelManager
from pipeline.runner import run_pipeline


if __name__ == "__main__":
    run_pipeline(SADAModelManager)
