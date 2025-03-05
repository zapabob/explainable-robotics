"""
変換ユーティリティモジュール

このモジュールはテンソルと配列の変換に関するユーティリティ関数を提供します。
"""

import numpy as np
import torch
from typing import Dict, List, Union, Any, Tuple, Optional

def tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    PyTorchテンソルをNumPy配列に変換
    
    Args:
        tensor: 変換するテンソル
        
    Returns:
        変換されたNumPy配列
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    
    if isinstance(tensor, torch.Tensor):
        # GPU上のテンソルの場合はCPUに移動
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        # 勾配を持つテンソルの場合はデタッチ
        if tensor.requires_grad:
            tensor = tensor.detach()
        
        return tensor.numpy()
    
    raise TypeError(f"変換できない型です: {type(tensor)}")

def numpy_to_tensor(
    array: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False
) -> torch.Tensor:
    """
    NumPy配列をPyTorchテンソルに変換
    
    Args:
        array: 変換する配列
        device: テンソルを配置するデバイス
        dtype: テンソルのデータ型
        requires_grad: 勾配計算を有効にするかどうか
        
    Returns:
        変換されたテンソル
    """
    if isinstance(array, torch.Tensor):
        tensor = array
        
        # デバイスの更新
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        
        # データ型の更新
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        
        # 勾配設定の更新
        if requires_grad and not tensor.requires_grad:
            tensor = tensor.requires_grad_(True)
        
        return tensor
    
    if isinstance(array, np.ndarray):
        # NumPy配列からテンソルを作成
        tensor = torch.from_numpy(array)
        
        # デバイスとデータ型の設定
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        
        # 勾配設定
        if requires_grad:
            tensor = tensor.requires_grad_(True)
        
        return tensor
    
    # その他の型の場合
    return torch.tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

def to_float_tensor(
    data: Union[np.ndarray, List, float, torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    様々なデータ形式を浮動小数点テンソルに変換
    
    Args:
        data: 変換するデータ
        device: テンソルを配置するデバイス
        
    Returns:
        浮動小数点テンソル
    """
    if isinstance(data, torch.Tensor):
        tensor = data
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
    else:
        tensor = torch.tensor(data, dtype=torch.float32)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor

def dict_to_tensors(
    data_dict: Dict[str, Any],
    device: Optional[torch.device] = None,
    float_only: bool = True
) -> Dict[str, torch.Tensor]:
    """
    辞書内の数値データをテンソルに変換
    
    Args:
        data_dict: 変換する辞書
        device: テンソルを配置するデバイス
        float_only: 浮動小数点値のみを変換するかどうか
        
    Returns:
        テンソルを含む新しい辞書
    """
    result = {}
    for key, value in data_dict.items():
        # 再帰的に処理（ネストされた辞書の場合）
        if isinstance(value, dict):
            result[key] = dict_to_tensors(value, device, float_only)
            continue
        
        # リストやタプルの場合は再帰的に処理
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], dict):
            result[key] = [dict_to_tensors(item, device, float_only) for item in value]
            continue
        
        # 数値データを変換
        try:
            # 数値データのみ変換
            is_numeric = isinstance(value, (int, float, np.number, np.ndarray, list, torch.Tensor))
            if is_numeric:
                # 浮動小数点フラグがオンで、かつ整数データの場合はスキップ
                if float_only and isinstance(value, (int, np.integer)):
                    result[key] = value
                else:
                    result[key] = to_float_tensor(value, device)
            else:
                result[key] = value
        except:
            # 変換できない場合は元の値をそのまま使用
            result[key] = value
    
    return result 