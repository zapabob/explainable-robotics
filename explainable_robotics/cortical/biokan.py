"""BioKAN - Bio-inspired Kolmogorov-Arnold Network

大脳皮質の6層構造を模倣した階層型ニューラルネットワークです。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

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
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-5) + self.beta

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
            layer_type: 層のタイプ ('sensory', 'association', 'motor')
            dropout: ドロップアウト率
        """
        super().__init__()
        
        self.layer_type = layer_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 主要な全結合層
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 層タイプに基づいた特殊化
        if layer_type == 'sensory':
            # 感覚層は入力の特徴抽出に特化
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            )
        elif layer_type == 'association':
            # 連合層は複雑な情報処理に特化
            self.lateral = nn.Linear(output_dim, output_dim)
            self.recurrent = nn.GRUCell(output_dim, output_dim)
            self.state = None
        elif layer_type == 'motor':
            # 運動層は出力の制御に特化
            self.gate = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.Sigmoid()
            )
        
        # 共通コンポーネント
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル [batch_size, input_dim]
            
        Returns:
            出力テンソル [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        if self.layer_type == 'sensory':
            # 感覚層: 注意機構を適用
            attention_weights = self.attention(x)
            x = x * attention_weights
        
        # 主要な変換
        out = self.linear(x)
        
        if self.layer_type == 'association':
            # 連合層: 横方向と再帰的接続
            lateral = self.lateral(out)
            
            # 状態の初期化（必要な場合）
            if self.state is None or self.state.size(0) != batch_size:
                self.state = torch.zeros(batch_size, self.output_dim, 
                                        device=x.device)
            
            # 再帰的更新
            self.state = self.recurrent(out, self.state)
            out = out + 0.1 * lateral + 0.1 * self.state
            
        elif self.layer_type == 'motor':
            # 運動層: ゲート機構
            gate_values = self.gate(x)
            out = out * gate_values
        
        # 共通処理
        out = self.activation(out)
        out = self.dropout(out)
        out = self.norm(out)
        
        return out
    
    def reset_state(self):
        """内部状態をリセット"""
        self.state = None


class BioKAN(nn.Module):
    """Bio-inspired Kolmogorov-Arnold Network"""
    
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
            num_layers: 層の数（デフォルト: 6、大脳皮質の6層構造に対応）
            dropout: ドロップアウト率
            learning_rate: 学習率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 神経伝達物質レベル（学習と適応のパラメータ）
        self.neurotransmitter_levels = {
            'dopamine': torch.tensor(0.5),  # 報酬と探索のバランス
            'serotonin': torch.tensor(0.5),  # 感情的安定性と回復力
            'norepinephrine': torch.tensor(0.5),  # 注意と警戒
            'acetylcholine': torch.tensor(0.5),  # 学習と記憶
            'gaba': torch.tensor(0.5),  # 抑制と制御
            'glutamate': torch.tensor(0.5)   # 興奮と可塑性
        }
        
        # レイヤーの作成
        self.layers = nn.ModuleList()
        
        # 入力層（感覚層）
        self.layers.append(CorticalLayer(
            input_dim, hidden_dim, 'sensory', dropout
        ))
        
        # 中間層（連合層）
        for i in range(num_layers - 2):
            self.layers.append(CorticalLayer(
                hidden_dim, hidden_dim, 'association', dropout
            ))
        
        # 出力層（運動層）
        self.layers.append(CorticalLayer(
            hidden_dim, output_dim, 'motor', dropout
        ))
        
        # 学習関連のパラメータ
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 学習の履歴
        self.training_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル [batch_size, input_dim]
            
        Returns:
            出力テンソル [batch_size, output_dim]
        """
        # 神経伝達物質による調整
        dopamine = self.neurotransmitter_levels['dopamine'].item()
        serotonin = self.neurotransmitter_levels['serotonin'].item()
        norepinephrine = self.neurotransmitter_levels['norepinephrine'].item()
        
        # 各層の適用
        for i, layer in enumerate(self.layers):
            # 層タイプに基づく調整
            if i == 0:  # 感覚層
                # ノルエピネフリンは感覚入力の注意レベルを調整
                attention_boost = 1.0 + 0.2 * (norepinephrine - 0.5)
                x = x * attention_boost
            
            # ドーパミンとセロトニンは情報処理の可塑性と安定性に影響
            if layer.layer_type == 'association':
                # 高いドーパミンは可塑性を増加
                # 高いセロトニンは処理の安定性を増加
                plasticity = 1.0 + 0.3 * (dopamine - 0.5)
                stability = 1.0 + 0.3 * (serotonin - 0.5)
                
                # 層に適用
                layer.dropout.p = max(0.05, min(0.5, 0.1 / stability))
            
            # 層を適用
            x = layer(x)
        
        return x
    
    def update(self, reward: torch.Tensor):
        """
        報酬に基づく更新
        
        Args:
            reward: 報酬値
        """
        # 神経伝達物質レベルの更新
        reward_val = reward.item()
        
        # ドーパミン: 報酬によって変動
        self.neurotransmitter_levels['dopamine'] = torch.clamp(
            self.neurotransmitter_levels['dopamine'] + 0.1 * reward_val,
            0.1, 0.9
        )
        
        # セロトニン: 長期的な報酬履歴に基づいて安定
        if len(self.training_history) > 0:
            avg_reward = sum(self.training_history[-10:]) / min(10, len(self.training_history))
            if reward_val > avg_reward:
                self.neurotransmitter_levels['serotonin'] = torch.clamp(
                    self.neurotransmitter_levels['serotonin'] + 0.05,
                    0.1, 0.9
                )
            else:
                self.neurotransmitter_levels['serotonin'] = torch.clamp(
                    self.neurotransmitter_levels['serotonin'] - 0.05,
                    0.1, 0.9
                )
        
        # 報酬履歴に追加
        self.training_history.append(reward_val)
        
        # 学習率の調整
        self.scheduler.step(reward)
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        トレーニングステップの実行
        
        Args:
            x: 入力データ
            y: 目標出力
            
        Returns:
            損失値
        """
        self.optimizer.zero_grad()
        output = self(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        self.optimizer.step()
        
        # 損失に基づく神経伝達物質の調整
        loss_val = loss.item()
        self.neurotransmitter_levels['glutamate'] = torch.clamp(
            self.neurotransmitter_levels['glutamate'] - 0.1 * loss_val,
            0.1, 0.9
        )
        self.neurotransmitter_levels['gaba'] = torch.clamp(
            self.neurotransmitter_levels['gaba'] + 0.1 * loss_val,
            0.1, 0.9
        )
        
        return loss_val
    
    def get_neurotransmitter_levels(self) -> Dict[str, float]:
        """
        現在の神経伝達物質レベルを取得
        
        Returns:
            神経伝達物質レベルの辞書
        """
        return {k: v.item() for k, v in self.neurotransmitter_levels.items()}
    
    def set_neurotransmitter_levels(self, levels: Dict[str, float]):
        """
        神経伝達物質レベルを設定
        
        Args:
            levels: 神経伝達物質レベルの辞書
        """
        for k, v in levels.items():
            if k in self.neurotransmitter_levels:
                self.neurotransmitter_levels[k] = torch.tensor(v).float()
    
    def reset(self):
        """モデルの状態をリセット"""
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                layer.reset_state()
        
        # 神経伝達物質を初期値にリセット
        for k in self.neurotransmitter_levels:
            self.neurotransmitter_levels[k] = torch.tensor(0.5)
        
        # 学習履歴をクリア
        self.training_history = []
    
    def save(self, path: str):
        """
        モデルの保存
        
        Args:
            path: 保存パス
        """
        torch.save({
            'state_dict': self.state_dict(),
            'neurotransmitter_levels': self.neurotransmitter_levels,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim
            }
        }, path)
        logger.info(f"モデルを保存しました: {path}")
    
    def load(self, path: str):
        """
        モデルの読み込み
        
        Args:
            path: 読み込みパス
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        self.neurotransmitter_levels = checkpoint['neurotransmitter_levels']
        logger.info(f"モデルを読み込みました: {path}") 