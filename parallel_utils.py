import torch
import torch.nn as nn

def setup_data_parallel(model, gpu_ids):
    """
    指定されたGPU IDのリストに基づいて、モデルをDataParallelでラップします。
    GPUが2台以上指定されている場合のみDataParallelを適用し、
    それ以外の場合はそのままモデルを返します。

    Args:
        model (torch.nn.Module): ラップするモデル
        gpu_ids (list of int): 使用するGPUのIDリスト

    Returns:
        torch.nn.Module: DataParallelでラップされたモデルまたは元のモデル
    """
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"Model is wrapped with DataParallel on GPUs: {gpu_ids}")
    else:
        print("DataParallel is not applied (single GPU or no GPU available)")
    return model