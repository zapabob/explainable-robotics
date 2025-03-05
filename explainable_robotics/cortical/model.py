"""
大脳皮質層構造を模倣したニューラルネットワークモデル

このモジュールは大脳皮質の6層構造を模倣した神経ネットワークモデルを実装します。
各層は実際の大脳皮質の機能特性を反映し、BioKANフレームワークと統合されています。
出力範囲は-1.0から1.0で、ヒューマノイドロボットの制御に適しています。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys
import json

# 相対インポートへ変更
from ..utils.logging import get_logger

# BioKANからの必要なインポート（プロジェクト構造に依存）
try:
    from kan.core.bio_kan import BioKAN
    from kan.core.glia.astrocyte import Astrocyte
    from kan.core.glia.microglia import Microglia
    from kan.core.neuromodulators.dopamine import DopamineSystem
    from kan.core.neuromodulators.acetylcholine import AcetylcholineSystem
    BIOKAN_AVAILABLE = True
except ImportError:
    print("警告: BioKANライブラリが見つかりません。モックを使用します。")
    BIOKAN_AVAILABLE = False
    # モッククラス定義
    class BioKAN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return torch.tanh(self.linear(x))
    
    class Astrocyte:
        def __init__(self, dim=None):
            pass
        def process(self, x):
            return x
    
    class Microglia:
        def __init__(self):
            pass
        def process(self, x):
            return x
    
    class DopamineSystem:
        def __init__(self):
            pass
    
    class AcetylcholineSystem:
        def __init__(self):
            pass

# Genesisライブラリとの統合
try:
    import genesis
    import genesis.motor as gmotor
    from genesis.neurotransmitters import NeurotransmitterSystem
    GENESIS_AVAILABLE = True
except ImportError:
    print("警告: Genesisライブラリが見つかりません。モックを使用します。")
    GENESIS_AVAILABLE = False
    # モック定義
    class NeurotransmitterSystem:
        def __init__(self):
            pass
        def set_level(self, transmitter_type, level, target_regions=None):
            pass
        def get_level(self, transmitter_type):
            return 0.5

# ロガーの取得
logger = get_logger(__name__)

class CorticalLayer(nn.Module):
    """
    大脳皮質の単一層を表現するニューラルネットワーク層
    
    各層は以下の特性を持ちます:
    - 層特有の変換（線形変換 + 非線形活性化）
    - 層特有の結合パターン
    - グリア細胞による調整
    - 神経伝達物質による調整
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_type: str,
        use_glia: bool = True,
        use_neuromodulation: bool = True,
        dropout: float = 0.1
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            output_dim: 出力次元
            layer_type: 層タイプ ('L1', 'L2', 'L3', 'L4', 'L5', 'L6' のいずれか)
            use_glia: グリア細胞を使用するかどうか
            use_neuromodulation: 神経伝達物質調整を使用するかどうか
            dropout: ドロップアウト率
        """
        super().__init__()
        self.layer_type = layer_type
        self.use_glia = use_glia
        self.use_neuromodulation = use_neuromodulation
        
        # 基本的な線形変換
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 層タイプに応じた特別な処理
        if layer_type == 'L1':
            # 層1（分子層）: 外部からの入力を受け取り、感覚情報を統合
            self.activation = nn.Tanh()  # 双極性活性化関数
            self.recurrent = None
        elif layer_type == 'L2' or layer_type == 'L3':
            # 層2/3（外錐体細胞層）: 皮質内結合と抽象化
            self.activation = nn.LeakyReLU(0.1)
            self.recurrent = nn.Linear(output_dim, output_dim)
        elif layer_type == 'L4':
            # 層4（内顆粒層）: 視床からの入力を受け取る
            self.activation = nn.ReLU()
            self.recurrent = None
        elif layer_type == 'L5':
            # 層5（内錐体細胞層）: 主要な出力層
            self.activation = nn.Tanh()  # 双極性活性化関数（-1〜1の出力）
            self.recurrent = nn.Linear(output_dim, output_dim)
        elif layer_type == 'L6':
            # 層6（多形細胞層）: 視床へのフィードバック
            self.activation = nn.Sigmoid()
            self.recurrent = nn.Linear(output_dim, output_dim)
        else:
            raise ValueError(f"不明な層タイプ: {layer_type}")
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
        
        # グリア細胞
        if use_glia:
            self.astrocyte = Astrocyte(output_dim)
            self.microglia = Microglia()
        
        # 神経調節物質
        if use_neuromodulation:
            self.dopamine_system = DopamineSystem()
            self.acetylcholine_system = AcetylcholineSystem()
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            state: 前の時間ステップの状態（再帰層用）
            context: 追加のコンテキスト情報
            
        Returns:
            output: 層の出力
            new_state: 更新された状態
        """
        # 基本的な線形変換
        output = self.linear(x)
        
        # 再帰的な処理（ある場合）
        if self.recurrent is not None and state is not None:
            recurrent_out = self.recurrent(state)
            output = output + recurrent_out
        
        # 神経調節物質の影響を適用
        if self.use_neuromodulation and context is not None:
            dopamine_level = context.get('dopamine', 0.5)
            acetylcholine_level = context.get('acetylcholine', 0.5)
            
            # ドーパミンは活性化関数の傾きを調整
            if self.layer_type in ['L2', 'L3', 'L5']:
                dopamine_factor = 0.5 + dopamine_level
                output = output * dopamine_factor
            
            # アセチルコリンは注意と精度を調整
            if self.layer_type in ['L1', 'L4']:
                acetylcholine_factor = 0.5 + acetylcholine_level
                output = output * acetylcholine_factor
        
        # 活性化関数
        output = self.activation(output)
        
        # ドロップアウト
        output = self.dropout(output)
        
        # グリア細胞の影響を適用
        if self.use_glia:
            # アストロサイトはシナプス結合を強化
            if hasattr(self, 'astrocyte'):
                output = self.astrocyte.process(output)
            
            # ミクログリアはノイズを低減
            if hasattr(self, 'microglia'):
                output = self.microglia.process(output)
        
        # 出力範囲を-1〜1に制限（必要な層のみ）
        if self.layer_type in ['L5', 'L1']:
            output = torch.tanh(output)  # 既にtanhを使用している場合は冗長ですが、安全のため
        
        return output, output  # 出力と新しい状態を返す

class CorticalModel(nn.Module):
    """
    完全な大脳皮質層構造を実装したニューラルネットワークモデル
    
    実際の大脳皮質の6層構造を模倣し、層間の正確な接続パターンを実装しています。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_glia: bool = True,
        use_neuromodulation: bool = True,
        dropout: float = 0.1
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ層の次元
            output_dim: 出力次元
            use_glia: グリア細胞を使用するかどうか
            use_neuromodulation: 神経伝達物質調整を使用するかどうか
            dropout: ドロップアウト率
        """
        super().__init__()
        self.use_glia = use_glia
        self.use_neuromodulation = use_neuromodulation
        
        # オプティマイザーの初期化
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # 大脳皮質の6層構造
        self.layer1 = CorticalLayer(input_dim, hidden_dim, 'L1', 
                                     use_glia, use_neuromodulation, dropout)
        self.layer2_3 = CorticalLayer(hidden_dim, hidden_dim, 'L2', 
                                      use_glia, use_neuromodulation, dropout)
        self.layer4 = CorticalLayer(hidden_dim, hidden_dim, 'L4', 
                                    use_glia, use_neuromodulation, dropout)
        self.layer5 = CorticalLayer(hidden_dim, output_dim, 'L5', 
                                    use_glia, use_neuromodulation, dropout)
        self.layer6 = CorticalLayer(output_dim, hidden_dim, 'L6', 
                                    use_glia, use_neuromodulation, dropout)
        
        # 層間のスキップ接続
        self.skip_2_5 = nn.Linear(hidden_dim, output_dim)
        self.skip_4_6 = nn.Linear(hidden_dim, hidden_dim)
        
        # バイオマーカーとして使用される状態値
        self.states = {
            'L1': None,
            'L2_3': None,
            'L4': None,
            'L5': None,
            'L6': None
        }
        
        # 神経伝達物質システム
        self.nt_system = NeurotransmitterSystem() if GENESIS_AVAILABLE else None
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            context: 追加のコンテキスト情報（神経伝達物質レベルなど）
            
        Returns:
            output: モデル出力（-1〜1の範囲）
        """
        # デフォルトのコンテキスト
        if context is None:
            context = {
                'dopamine': 0.5,
                'acetylcholine': 0.5,
                'serotonin': 0.5,
                'noradrenaline': 0.5
            }
        
        # 層1: 外部からの入力を処理
        l1_out, l1_state = self.layer1(x, self.states['L1'], context)
        self.states['L1'] = l1_state
        
        # 層2/3: 皮質内統合
        l2_3_out, l2_3_state = self.layer2_3(l1_out, self.states['L2_3'], context)
        self.states['L2_3'] = l2_3_state
        
        # 層4: 視床入力の処理
        l4_out, l4_state = self.layer4(l1_out, self.states['L4'], context)  # L1から直接入力
        self.states['L4'] = l4_state
        
        # 層間のスキップ接続
        skip_2_5_out = self.skip_2_5(l2_3_out)
        skip_4_6_out = self.skip_4_6(l4_out)
        
        # 層5: 主要出力層
        l5_input = l4_out + skip_2_5_out  # 層4からの入力と層2/3からのスキップ接続
        l5_out, l5_state = self.layer5(l5_input, self.states['L5'], context)
        self.states['L5'] = l5_state
        
        # 層6: 視床へのフィードバック
        l6_input = l5_out + skip_4_6_out  # 層5からの入力と層4からのスキップ接続
        l6_out, l6_state = self.layer6(l6_input, self.states['L6'], context)
        self.states['L6'] = l6_state
        
        # 最終出力は層5から（運動出力）
        # 出力範囲を-1〜1に制限
        output = torch.tanh(l5_out)
        
        return output
    
    def modulate_neurotransmitter(
        self,
        transmitter_type: str,
        level: float,
        target_regions: Optional[List[str]] = None
    ) -> None:
        """
        神経伝達物質レベルを調整
        
        Args:
            transmitter_type: 神経伝達物質タイプ
            level: 新しいレベル値（0〜1）
            target_regions: 影響を受ける領域（Noneの場合はすべての領域）
        """
        if self.nt_system is not None:
            self.nt_system.set_level(transmitter_type, level, target_regions)
            logger.info(f"神経伝達物質 {transmitter_type} のレベルを {level} に調整しました")
    
    def reset_states(self):
        """内部状態をリセット"""
        self.states = {
            'L1': None,
            'L2_3': None,
            'L4': None,
            'L5': None,
            'L6': None
        }
        logger.debug("内部状態をリセットしました")
    
    def explain_activation(self, layer_name: str) -> Dict[str, Any]:
        """
        指定された層の活性化を説明
        
        Args:
            layer_name: 説明する層の名前
            
        Returns:
            説明情報を含む辞書
        """
        explanations = {
            'L1': "層1（分子層）は外部入力の初期処理を担当しています。現在の活性化パターンは感覚入力の統合を示しています。",
            'L2_3': "層2/3（外錐体細胞層）は皮質内処理と抽象化を担当しています。現在のパターンは特徴抽出と空間的統合を示しています。",
            'L4': "層4（内顆粒層）は視床からの入力を処理しています。現在の活性化は感覚情報の初期分類を示しています。",
            'L5': "層5（内錐体細胞層）は主要な出力層です。現在のパターンは実行される運動コマンドを示しています。",
            'L6': "層6（多形細胞層）は視床へのフィードバックを提供しています。現在の活性化は注意の調整と情報フィルタリングを示しています。"
        }
        
        # 層の状態値
        state_value = self.states.get(layer_name, None)
        if state_value is not None:
            # 実際の値を含む説明を作成
            state_stats = {
                'mean': float(torch.mean(state_value).item()),
                'std': float(torch.std(state_value).item()),
                'min': float(torch.min(state_value).item()),
                'max': float(torch.max(state_value).item()),
            }
            
            explanation = {
                'description': explanations.get(layer_name, "情報なし"),
                'statistics': state_stats,
                'interpretation': self._interpret_activation(layer_name, state_stats)
            }
            return explanation
        
        return {'description': "この層の活性化情報はありません。"}
    
    def _interpret_activation(self, layer_name: str, stats: Dict[str, float]) -> str:
        """活性化統計の解釈を生成"""
        mean, std = stats['mean'], stats['std']
        
        if layer_name == 'L5':  # 出力層
            if mean > 0.6:
                return "強い順方向の運動コマンド"
            elif mean < -0.6:
                return "強い逆方向の運動コマンド"
            elif -0.2 <= mean <= 0.2 and std < 0.3:
                return "静止状態またはバランス状態"
            else:
                return "中程度の運動コマンド"
        elif layer_name == 'L1':  # 入力層
            if std > 0.5:
                return "多様な感覚入力を処理中"
            else:
                return "一貫した感覚入力を処理中"
        else:
            if std > 0.7:
                return "高度に差別化された活性化パターン"
            elif std < 0.2:
                return "一様な活性化パターン"
            else:
                return "バランスの取れた活性化パターン"

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        モデルの損失関数を計算
        
        Args:
            x: 入力テンソル
            y: 正解ラベル
            
        Returns:
            loss: 計算された損失
        """
        output = self(x)
        loss = F.mse_loss(output, y)
        return loss

class CorticalBioKAN(nn.Module):
    """
    大脳皮質モデルとBioKANを統合したハイブリッドモデル
    
    大脳皮質の層構造によるモータ制御と、BioKANによる学習と最適化を組み合わせます。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        kan_hidden_dim: int = 64,
        use_glia: bool = True,
        use_neuromodulation: bool = True
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 皮質モデルの隠れ層次元
            output_dim: 出力次元
            kan_hidden_dim: BioKANの隠れ層次元
            use_glia: グリア細胞を使用するかどうか
            use_neuromodulation: 神経伝達物質調整を使用するかどうか
        """
        super().__init__()
        
        # 皮質モデルの初期化
        self.cortical_model = CorticalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_glia=use_glia,
            use_neuromodulation=use_neuromodulation
        )
        
        # BioKANモデルの初期化
        if BIOKAN_AVAILABLE:
            self.biokan = BioKAN(
                input_dim=input_dim,
                hidden_dim=kan_hidden_dim,
                output_dim=output_dim,
                use_neuromodulation=use_neuromodulation,
                use_glia=use_glia
            )
        else:
            # BioKANが利用できない場合のフォールバック
            self.biokan = nn.Sequential(
                nn.Linear(input_dim, kan_hidden_dim),
                nn.ReLU(),
                nn.Linear(kan_hidden_dim, output_dim),
                nn.Tanh()
            )
            logger.warning("BioKANモデルが利用できないため、標準的なMLPモデルにフォールバックしました")
        
        # 出力統合のための変換
        self.integration_layer = nn.Linear(output_dim * 2, output_dim)
        
        # 説明結果のキャッシュ
        self.explanation_cache = {}
    
    def forward(
        self, 
        x: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            context: 追加のコンテキスト情報
            
        Returns:
            output: モデル出力（-1〜1の範囲）
        """
        # 皮質モデルからの出力
        cortical_output = self.cortical_model(x, context)
        
        # BioKANからの出力
        biokan_output = self.biokan(x)
        
        # 出力の結合と統合
        combined_output = torch.cat([cortical_output, biokan_output], dim=1)
        integrated_output = self.integration_layer(combined_output)
        
        # 最終出力は-1〜1の範囲に制限
        final_output = torch.tanh(integrated_output)
        
        return final_output
    
    def modulate_neurotransmitter(
        self,
        transmitter_type: str,
        level: float,
        target_regions: Optional[List[str]] = None
    ) -> None:
        """
        神経伝達物質レベルを調整（両方のモデルに適用）
        
        Args:
            transmitter_type: 神経伝達物質タイプ
            level: 新しいレベル値（0〜1）
            target_regions: 影響を受ける領域（Noneの場合はすべての領域）
        """
        # 皮質モデルへの適用
        self.cortical_model.modulate_neurotransmitter(transmitter_type, level, target_regions)
        
        # BioKANへの適用（互換性がある場合）
        if hasattr(self.biokan, 'modulate_neurotransmitter'):
            self.biokan.modulate_neurotransmitter(transmitter_type, level, target_regions)
    
    def explain_action(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        現在の行動を説明
        
        Args:
            input_data: 説明する入力データ
            
        Returns:
            説明情報を含む辞書
        """
        # 入力に基づいて出力を計算
        with torch.no_grad():
            output = self(input_data)
            
            # 皮質モデルの層ごとの活性化を説明
            layer_explanations = {}
            for layer_name in ['L1', 'L2_3', 'L4', 'L5', 'L6']:
                layer_explanations[layer_name] = self.cortical_model.explain_activation(layer_name)
            
            # 出力値の解釈
            output_np = output.numpy() if isinstance(output, torch.Tensor) else output
            
            explanation = {
                'cortical_layers': layer_explanations,
                'output_value': float(output_np.mean()),
                'output_interpretation': self._interpret_output(output_np),
                'confidence': self._calculate_confidence(layer_explanations)
            }
            
            # 説明のキャッシュを更新
            self.explanation_cache = explanation
            
            return explanation
    
    def _interpret_output(self, output: np.ndarray) -> str:
        """出力値の自然言語解釈を生成"""
        mean_value = float(np.mean(output))
        
        if mean_value > 0.8:
            return "強い前進動作"
        elif mean_value > 0.4:
            return "中程度の前進動作"
        elif mean_value > 0.1:
            return "弱い前進動作"
        elif mean_value > -0.1:
            return "ほぼ静止状態"
        elif mean_value > -0.4:
            return "弱い後退動作"
        elif mean_value > -0.8:
            return "中程度の後退動作"
        else:
            return "強い後退動作"
    
    def _calculate_confidence(self, layer_explanations: Dict[str, Dict[str, Any]]) -> float:
        """モデルの確信度を計算"""
        # 出力層（L5）の標準偏差に基づく確信度
        l5_stats = layer_explanations.get('L5', {}).get('statistics', {})
        std = l5_stats.get('std', 0.5)
        
        # 標準偏差が低いほど確信度が高い
        confidence = max(0.0, 1.0 - std * 2)
        return float(confidence)

def create_cortical_model(
    input_features: int,
    motor_outputs: int,
    hidden_size: int = 128,
    use_biologically_inspired: bool = True
) -> Union[CorticalBioKAN, CorticalModel]:
    """
    ヒューマノイドロボット制御用の皮質モデルを作成
    
    Args:
        input_features: 入力特徴量の数
        motor_outputs: モーター出力の数
        hidden_size: 隠れ層のサイズ
        use_biologically_inspired: 生物学的に着想を得たモデルを使用するかどうか
        
    Returns:
        作成されたモデル
    """
    logger.info(f"皮質モデルを作成: 入力={input_features}, 出力={motor_outputs}, 生物学的={use_biologically_inspired}")
    
    if use_biologically_inspired:
        return CorticalBioKAN(
            input_dim=input_features,
            hidden_dim=hidden_size,
            output_dim=motor_outputs,
            use_glia=True,
            use_neuromodulation=True
        )
    else:
        return CorticalModel(
            input_dim=input_features,
            hidden_dim=hidden_size,
            output_dim=motor_outputs,
            use_glia=False,
            use_neuromodulation=False
        )

def save_model_explanation(model: Union[CorticalBioKAN, CorticalModel], filename: str) -> None:
    """
    モデルの説明をJSONファイルに保存
    
    Args:
        model: 説明するモデル
        filename: 保存先のファイル名
    """
    explanation = {
        'model_type': model.__class__.__name__,
        'uses_glia': getattr(model, 'use_glia', False),
        'uses_neuromodulation': getattr(model, 'use_neuromodulation', False),
        'layer_structure': {
            'L1': "分子層 - 外部入力の初期処理",
            'L2_3': "外錐体細胞層 - 皮質内処理と抽象化",
            'L4': "内顆粒層 - 視床入力の処理",
            'L5': "内錐体細胞層 - 主要な出力層",
            'L6': "多形細胞層 - 視床へのフィードバック"
        },
        'activation_functions': {
            'L1': "双極性（Tanh）- 感覚情報の相対的強度を保持",
            'L2_3': "LeakyReLU - 弱い信号も伝播可能",
            'L4': "ReLU - スパースコーディングを促進",
            'L5': "双極性（Tanh）- モーター出力の方向と強度を表現",
            'L6': "Sigmoid - フィードバック強度の調整"
        },
        'output_range': "出力は-1.0から1.0の範囲。これにより、モーターに対する反対方向の制御が可能になります。"
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)
    
    logger.info(f"モデル説明を保存しました: {filename}") 