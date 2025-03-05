"""
大脳皮質モデル（Cortical Models）モジュール

大脳皮質の層構造を模倣したニューラルネットワークモデルの実装を提供します。
"""

from ..utils.logging import get_logger
from .model import CorticalLayer, CorticalModel, CorticalBioKAN, create_cortical_model
from .layers import LayerType, ConnectionPattern, ActivationType
from .biokan import BioKAN, LayerNorm

# ロガーの初期化
logger = get_logger(__name__)

__all__ = ["BioKAN", "CorticalLayer", "LayerNorm"] 