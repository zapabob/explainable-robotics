"""BioKAN - Bio-inspired Kolmogorov-Arnold Network

大脳皮質の6層構造を模倣した階層型ニューラルネットワークです。
三値入力（抑制:-1、中立:0、興奮:1）を処理し、神経科学的に妥当なモデルを実装しています。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import os
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)

class LayerNorm(nn.Module):
    """カスタム層正規化"""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-5
        return self.gamma * (x - mean) / std + self.beta

class CorticalLayer(nn.Module):
    """大脳皮質層を模倣したニューラルネットワーク層"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_type: str,
        dropout: float = 0.1
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            output_dim: 出力次元
            layer_type: 層のタイプ ('sensory', 'association', 'motor', 'inhibitory', 'excitatory')
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.layer_type = layer_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 層の種類に基づく内部パラメータの設定
        if layer_type == 'sensory':
            # 感覚入力層（大脳皮質の層4に相当）- 視床からの入力を受け取る
            self.linear = nn.Linear(input_dim, output_dim)
            self.activation = nn.ReLU()
            self.norm = LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
            
            # 感覚入力のゲーティング（アセチルコリンによる変調）
            self.sensory_gate = nn.Parameter(torch.ones(1))
            
        elif layer_type == 'association':
            # 連合層（大脳皮質の層2/3に相当）- 横方向の接続
            self.linear1 = nn.Linear(input_dim, output_dim * 2)
            self.linear2 = nn.Linear(output_dim * 2, output_dim)
            self.activation = nn.ReLU()
            self.norm = LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)
            
            # 横方向抑制のためのパラメータ（GABA作用をモデル化）
            self.lateral_inhibition = nn.Parameter(torch.tensor(0.5))
            
        elif layer_type == 'motor':
            # 運動出力層（大脳皮質の層5に相当）- 脊髄や脳幹への出力
            self.linear = nn.Linear(input_dim, output_dim)
            self.activation = nn.Tanh()  # 運動出力は-1～1の範囲
            
            # 行動選択のためのドーパミン変調パラメータ
            self.action_bias = nn.Parameter(torch.zeros(output_dim))
            
        elif layer_type == 'inhibitory':
            # 抑制性ニューロン層（GABA作動性ニューロンをモデル化）
            self.linear = nn.Linear(input_dim, output_dim)
            self.activation = nn.Sigmoid()  # 0～1の抑制強度
            
            # GABA伝達を表すパラメータ
            self.inhibitory_strength = nn.Parameter(torch.tensor(0.7))
            
        elif layer_type == 'excitatory':
            # 興奮性ニューロン層（グルタミン酸作動性ニューロンをモデル化）
            self.linear = nn.Linear(input_dim, output_dim)
            self.activation = nn.ReLU()
            
            # グルタミン酸伝達を表すパラメータ
            self.excitatory_strength = nn.Parameter(torch.tensor(0.7))
            
        else:
            raise ValueError(f"不明な層タイプ: {layer_type}")
            
        # 神経伝達物質変調パラメータ
        self.nt_modulation = nn.ParameterDict({
            'dopamine': nn.Parameter(torch.tensor(0.5)),
            'serotonin': nn.Parameter(torch.tensor(0.5)),
            'acetylcholine': nn.Parameter(torch.tensor(0.5)),
            'noradrenaline': nn.Parameter(torch.tensor(0.5)),
            'glutamate': nn.Parameter(torch.tensor(0.5)),
            'gaba': nn.Parameter(torch.tensor(0.5))
        })
        
        # 層の内部状態
        self.state = torch.zeros(output_dim)
        
    def forward(self, x: torch.Tensor, nt_levels: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            nt_levels: 神経伝達物質レベル (オプション)
            
        Returns:
            出力テンソル
        """
        # 神経伝達物質レベルのデフォルト値
        if nt_levels is None:
            nt_levels = {
                'dopamine': 0.5,
                'serotonin': 0.5,
                'acetylcholine': 0.5,
                'noradrenaline': 0.5,
                'glutamate': 0.5,
                'gaba': 0.5
            }
            
        # 層の種類に応じた計算
        if self.layer_type == 'sensory':
            # アセチルコリンレベルによる感覚ゲーティング
            sensory_gate = torch.sigmoid(self.sensory_gate * nt_levels['acetylcholine'])
            x = x * sensory_gate
            
            out = self.linear(x)
            out = self.activation(out)
            out = self.norm(out)
            out = self.dropout(out)
            
        elif self.layer_type == 'association':
            # ノルアドレナリンによる信号対雑音比の調整
            noise_factor = 1.0 - nt_levels['noradrenaline']
            if noise_factor > 0:
                noise = torch.randn_like(x) * noise_factor * 0.1
                x = x + noise
                
            out = self.linear1(x)
            out = self.activation(out)
            
            # GABAによる側方抑制
            lateral_inhibition = self.lateral_inhibition * nt_levels['gaba']
            if lateral_inhibition > 0:
                # 各ニューロンの出力が他のニューロンを抑制
                inhibition = F.softmax(out, dim=-1) * lateral_inhibition
                out = out * (1 - inhibition)
                
            out = self.linear2(out)
            out = self.activation(out)
            out = self.norm(out)
            out = self.dropout(out)
            
        elif self.layer_type == 'motor':
            # ドーパミンによる行動バイアス
            action_bias = self.action_bias * nt_levels['dopamine']
            
            out = self.linear(x)
            out = out + action_bias
            out = self.activation(out)
            
            # セロトニンによる衝動性制御（活性化のスケーリング）
            impulsivity_control = 0.5 + 0.5 * nt_levels['serotonin']
            out = out * impulsivity_control
            
        elif self.layer_type == 'inhibitory':
            # GABA作動性ニューロンは抑制性出力を生成
            inhibitory_strength = self.inhibitory_strength * nt_levels['gaba']
            
            out = self.linear(x)
            out = self.activation(out)
            out = out * inhibitory_strength
            
            # 出力は抑制強度なので負の値に
            out = -1.0 * out
            
        elif self.layer_type == 'excitatory':
            # グルタミン酸作動性ニューロンは興奮性出力を生成
            excitatory_strength = self.excitatory_strength * nt_levels['glutamate']
            
            out = self.linear(x)
            out = self.activation(out)
            out = out * excitatory_strength
            
        # 層の内部状態を更新
        self.state = out.detach().clone()
        
        return out
        
    def reset_state(self):
        """層の内部状態をリセット"""
        self.state = torch.zeros(self.output_dim)

class TrinaryCorticalProcessor(nn.Module):
    """三値入力（抑制:-1、中立:0、興奮:1）を処理する大脳皮質モデル"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            output_dim: 出力次元
            hidden_dim: 隠れ層の次元
        """
        super().__init__()
        
        # 三値入力の処理のための特殊層
        self.inhibitory_layer = CorticalLayer(input_dim, hidden_dim, 'inhibitory')
        self.excitatory_layer = CorticalLayer(input_dim, hidden_dim, 'excitatory')
        
        # 抑制性信号と興奮性信号を統合
        self.integration_layer = CorticalLayer(hidden_dim * 2, output_dim, 'association')
        
    def forward(
        self,
        inhibitory_input: torch.Tensor,  # -1: 抑制
        neutral_input: torch.Tensor,     # 0: 中立
        excitatory_input: torch.Tensor,  # 1: 興奮
        nt_levels: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        三値入力を処理
        
        Args:
            inhibitory_input: 抑制入力マスク (-1)
            neutral_input: 中立入力マスク (0)
            excitatory_input: 興奮入力マスク (1)
            nt_levels: 神経伝達物質レベル
            
        Returns:
            出力テンソル
        """
        # 抑制信号と興奮信号を個別に処理
        inhibitory_output = self.inhibitory_layer(inhibitory_input, nt_levels)
        excitatory_output = self.excitatory_layer(excitatory_input, nt_levels)
        
        # 二つの出力を結合
        combined = torch.cat([inhibitory_output, excitatory_output], dim=-1)
        
        # 統合層で処理
        output = self.integration_layer(combined, nt_levels)
        
        return output

class BioKAN(nn.Module):
    """
    生物学的妥当性を持つKolmogorov-Arnold Network (BioKAN)
    
    大脳皮質の6層構造を模倣し、神経伝達物質システムを統合した
    コルモゴロフアーノルドネットワーク実装。
    KANは複雑な多変数関数を近似するためのネットワークであり、
    神経科学的妥当性を高めるために拡張されています。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 6,
        dropout: float = 0.1,
        learning_rate: float = 1e-3
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ層の次元
            output_dim: 出力次元
            num_layers: 層の数（デフォルト：6、大脳皮質の層数に対応）
            dropout: ドロップアウト率
            learning_rate: 学習率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # 三値入力プロセッサ
        self.trinary_processor = TrinaryCorticalProcessor(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim // 2
        )
        
        # 大脳皮質の層構造
        self.layers = nn.ModuleList()
        
        # 層1 (分子層、主に感覚入力を受け取る)
        self.layers.append(CorticalLayer(hidden_dim, hidden_dim, 'sensory', dropout))
        
        # 層2/3 (外錐体細胞層、皮質内の連合線維を形成)
        self.layers.append(CorticalLayer(hidden_dim, hidden_dim, 'association', dropout))
        
        # 層4 (内顆粒層、視床からの入力を受け取る)
        self.layers.append(CorticalLayer(hidden_dim, hidden_dim, 'sensory', dropout))
        
        # 層5 (内錐体細胞層、皮質下構造への主要な出力層)
        self.layers.append(CorticalLayer(hidden_dim, hidden_dim, 'motor', dropout))
        
        # 層6 (多形細胞層、視床への投射を提供)
        self.layers.append(CorticalLayer(hidden_dim, output_dim, 'motor', dropout))
        
        # 神経伝達物質レベル
        self.neurotransmitter_levels = {
            'dopamine': 0.5,      # 強化学習、行動選択、報酬予測
            'serotonin': 0.5,     # 気分、衝動制御、感情調節
            'acetylcholine': 0.5, # 注意、学習、記憶形成
            'noradrenaline': 0.5, # 覚醒、集中、警戒
            'glutamate': 0.5,     # 興奮性神経伝達
            'gaba': 0.5           # 抑制性神経伝達
        }
        
        # 最適化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 報酬予測誤差 (RPE) - ドーパミン変調用
        self.predicted_reward = torch.tensor(0.0)
        
        # 状態追跡
        self.last_output = None
        self.last_action = None
        
        logger.info(f"BioKAN initialized with {input_dim} inputs, {hidden_dim} hidden, {output_dim} outputs, {num_layers} layers")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播計算
        
        入力を大脳皮質層構造を通して処理し、出力を生成します。
        
        Args:
            x: 入力テンソル (バッチサイズ x 入力次元) または (入力次元)
            
        Returns:
            出力テンソル (バッチサイズ x 出力次元) または (出力次元)
        """
        # 入力の形状を確認し、必要に応じてバッチ次元を追加
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # (入力次元) -> (1 x 入力次元)
            
        # 連続値入力を三値入力に変換
        inhibitory_mask = (x < -0.3).float()  # -1: 抑制入力
        excitatory_mask = (x > 0.3).float()   # 1: 興奮入力
        neutral_mask = ((x >= -0.3) & (x <= 0.3)).float()  # 0: 中立入力
        
        # 三値入力プロセッサで処理
        x = self.trinary_processor(
            inhibitory_mask,
            neutral_mask,
            excitatory_mask,
            self.neurotransmitter_levels
        )
        
        # 各大脳皮質層を通過
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x, self.neurotransmitter_levels)
            layer_outputs.append(x)
            
        # 層間の横方向接続（スキップ接続）
        if len(self.layers) > 2:
            # 層2/3 -> 層5 接続 (フィードフォワード)
            layer_outputs[3] = layer_outputs[3] + 0.2 * layer_outputs[1]
            
            # 層5 -> 層2/3 接続 (フィードバック)
            if self.training:
                # 訓練時のみ更新（次回のフォワードパスに影響）
                layer_outputs[1] = layer_outputs[1] + 0.1 * layer_outputs[3].detach()
        
        # 最終出力を保存
        self.last_output = x.detach().clone()
        
        # バッチ次元が1の場合は削除
        if x.shape[0] == 1:
            x = x.squeeze(0)
            
        return x
        
    def update(self, reward: torch.Tensor):
        """
        報酬信号に基づいてモデルを更新
        
        Args:
            reward: 報酬値
        """
        if self.last_output is None:
            logger.warning("更新を実行できません: 前回の出力がありません")
            return
            
        # 報酬予測誤差の計算
        reward_prediction_error = reward - self.predicted_reward
        
        # ドーパミンレベルを報酬予測誤差に基づいて更新
        dopamine_delta = float(reward_prediction_error.item()) * 0.1
        self.neurotransmitter_levels['dopamine'] = max(0.0, min(1.0, 
            self.neurotransmitter_levels['dopamine'] + dopamine_delta
        ))
        
        # 予測報酬の更新
        self.predicted_reward = 0.9 * self.predicted_reward + 0.1 * reward
        
        # セロトニンレベルの調整（報酬と逆相関の傾向）
        if dopamine_delta > 0:
            # 報酬時はセロトニンを緩やかに減少
            self.neurotransmitter_levels['serotonin'] = max(0.0, min(1.0,
                self.neurotransmitter_levels['serotonin'] - 0.01
            ))
        else:
            # 報酬なしの場合はセロトニンを緩やかに増加
            self.neurotransmitter_levels['serotonin'] = max(0.0, min(1.0,
                self.neurotransmitter_levels['serotonin'] + 0.01
            ))
            
        # 神経伝達物質レベルをログ出力
        logger.debug(f"NT levels - DA: {self.neurotransmitter_levels['dopamine']:.2f}, "
                    f"5-HT: {self.neurotransmitter_levels['serotonin']:.2f}, "
                    f"ACh: {self.neurotransmitter_levels['acetylcholine']:.2f}, "
                    f"NE: {self.neurotransmitter_levels['noradrenaline']:.2f}")
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        1ステップの訓練を実行
        
        Args:
            x: 入力データ
            y: 目標出力
            
        Returns:
            損失値
        """
        self.train()
        self.optimizer.zero_grad()
        
        # 順伝播
        output = self(x)
        
        # 損失計算
        loss = F.mse_loss(output, y)
        
        # 逆伝播
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        # パラメータ更新
        self.optimizer.step()
        
        return loss.item()
        
    def process_trinary_input(
        self,
        inhibitory: torch.Tensor,
        neutral: torch.Tensor,
        excitatory: torch.Tensor
    ) -> torch.Tensor:
        """
        三値入力を直接処理
        
        Args:
            inhibitory: 抑制入力マスク (-1)
            neutral: 中立入力マスク (0)
            excitatory: 興奮入力マスク (1)
            
        Returns:
            処理後の出力
        """
        # バッチ次元の有無を確認
        has_batch = len(inhibitory.shape) > 1
        
        if not has_batch:
            inhibitory = inhibitory.unsqueeze(0)
            neutral = neutral.unsqueeze(0)
            excitatory = excitatory.unsqueeze(0)
            
        # 三値入力プロセッサで処理
        x = self.trinary_processor(
            inhibitory,
            neutral,
            excitatory,
            self.neurotransmitter_levels
        )
        
        # 各大脳皮質層を通過
        for layer in self.layers:
            x = layer(x, self.neurotransmitter_levels)
            
        # バッチ次元が1の場合は削除
        if not has_batch:
            x = x.squeeze(0)
            
        return x
        
    def get_neurotransmitter_levels(self) -> Dict[str, float]:
        """
        現在の神経伝達物質レベルを取得
        
        Returns:
            神経伝達物質レベルの辞書
        """
        return self.neurotransmitter_levels.copy()
        
    def set_neurotransmitter_levels(self, levels: Dict[str, float]):
        """
        神経伝達物質レベルを設定
        
        Args:
            levels: 神経伝達物質レベルの辞書
        """
        for key, value in levels.items():
            if key in self.neurotransmitter_levels:
                self.neurotransmitter_levels[key] = max(0.0, min(1.0, value))
        
        logger.debug(f"更新されたNTレベル: {self.neurotransmitter_levels}")
        
    def reset(self):
        """
        モデルの状態をリセット
        """
        # 神経伝達物質レベルをデフォルトに戻す
        self.neurotransmitter_levels = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'acetylcholine': 0.5,
            'noradrenaline': 0.5,
            'glutamate': 0.5,
            'gaba': 0.5
        }
        
        # 各層の状態をリセット
        for layer in self.layers:
            layer.reset_state()
            
        self.predicted_reward = torch.tensor(0.0)
        self.last_output = None
        self.last_action = None
        
        logger.info("BioKAN状態をリセットしました")
        
    def save(self, path: str):
        """
        モデルを保存
        
        Args:
            path: 保存先のパス
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        state = {
            'model_state': self.state_dict(),
            'neurotransmitter_levels': self.neurotransmitter_levels,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'learning_rate': self.learning_rate
            }
        }
        
        torch.save(state, path)
        logger.info(f"モデルを保存しました: {path}")
        
    def load(self, path: str):
        """
        モデルをロード
        
        Args:
            path: モデルファイルのパス
        """
        if not os.path.exists(path):
            logger.error(f"モデルファイルが見つかりません: {path}")
            return False
            
        try:
            state = torch.load(path)
            self.load_state_dict(state['model_state'])
            self.neurotransmitter_levels = state['neurotransmitter_levels']
            
            logger.info(f"モデルをロードしました: {path}")
            return True
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {str(e)}")
            return False 