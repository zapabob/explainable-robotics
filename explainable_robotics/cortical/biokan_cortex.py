"""
BioKANを大脳皮質として使用するモジュール

KANの生体模倣アーキテクチャを用いて大脳皮質の機能を模倣します。
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

try:
    from kan.core.bio_kan import BioKAN
    from kan.core.neuromodulators import (
        DopamineSystem, SerotoninSystem, AcetylcholineSystem,
        NoradrenalineSystem, GABASystem, GlutamateSystem
    )
    BIOKAN_AVAILABLE = True
except ImportError:
    print("警告: BioKANライブラリがインストールされていません。モックを使用します。")
    BIOKAN_AVAILABLE = False
    
    # モックBioKAN実装
    class BioKAN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim),
                torch.nn.Tanh()
            )
            
        def forward(self, x):
            return self.layers(x)
    
    class DopamineSystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            return x * (0.5 + self.level)
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))
    
    class SerotoninSystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            return x * (0.5 + self.level)
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))
    
    class AcetylcholineSystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            return x * (0.5 + self.level)
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))
    
    # 新しい神経伝達物質システムを追加
    class NoradrenalineSystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            # ノルアドレナリンは覚醒度と注意を制御 - 高いレベルは入力への感度を増加
            gain = 1.0 + self.level
            return x * gain
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))
    
    class GABASystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            # GABAは抑制性神経伝達物質 - 高いレベルは活性化を抑制
            inhibition = 1.0 - (self.level * 0.5)  # 最大50%の抑制
            return x * inhibition
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))
    
    class GlutamateSystem:
        def __init__(self):
            self.level = 0.5
            
        def modulate(self, x):
            # グルタミン酸は興奮性神経伝達物質 - 高いレベルは活性化を促進
            excitation = 1.0 + (self.level * 0.5)  # 最大50%の促進
            return x * excitation
            
        def set_level(self, level):
            self.level = max(0.0, min(1.0, level))

from ..utils.logging import get_logger

logger = get_logger(__name__)

class CorticalBioKAN:
    """
    BioKANを大脳皮質として使用するクラス
    
    大脳皮質の異なる層としてBioKANのネットワークを使用し、
    神経伝達物質による調節機能を備えています。
    """
    
    def __init__(
        self,
        input_dim: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 50,
        num_layers: int = 6,
        learning_rate: float = 0.001
    ):
        """
        初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ層の次元
            output_dim: 出力次元
            num_layers: 層の数（大脳皮質の6層構造を表現）
            learning_rate: 学習率
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 大脳皮質の層構造の作成
        self.cortical_layers = []
        
        # 入力層（Layer 1）
        self.layer1 = BioKAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            activation='tanh'
        )
        
        # 中間層（Layers 2-5）
        for i in range(1, num_layers - 1):
            layer = BioKAN(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                activation='relu' if i % 2 == 0 else 'tanh'
            )
            self.cortical_layers.append(layer)
        
        # 出力層（Layer 6）
        self.layer6 = BioKAN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation='tanh'
        )
        
        # 神経調節物質システム
        self.dopamine_system = DopamineSystem()
        self.serotonin_system = SerotoninSystem()
        self.acetylcholine_system = AcetylcholineSystem()
        self.noradrenaline_system = NoradrenalineSystem()
        self.gaba_system = GABASystem()
        self.glutamate_system = GlutamateSystem()
        
        # 最適化器
        all_params = (
            list(self.layer1.parameters()) +
            sum([list(layer.parameters()) for layer in self.cortical_layers], []) +
            list(self.layer6.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=learning_rate)
        
        logger.info(f"BioKAN大脳皮質モデルを作成しました（{num_layers}層）")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: 入力テンソル
            
        Returns:
            出力テンソル
        """
        # 入力層（Layer 1）- 感覚入力の処理
        h = self.layer1(x)
        
        # アセチルコリンによる調整（注意力と精度に影響）
        h = self.acetylcholine_system.modulate(h)
        
        # ノルアドレナリンによる調整（覚醒と注意力に影響）
        h = self.noradrenaline_system.modulate(h)
        
        # 中間層の処理
        for i, layer in enumerate(self.cortical_layers):
            # Layer 2 - パターン検出と統合
            if i == 0:
                # グルタミン酸による調整（興奮性制御）
                h = self.glutamate_system.modulate(h)
            
            # Layer 3 - 抽象的な情報処理
            elif i == 1:  
                # ドーパミンによる調整（報酬学習に影響）
                h = self.dopamine_system.modulate(h)
            
            # Layer 4 - 感覚情報の詳細処理
            elif i == 2:  
                # セロトニンによる調整（感情と気分に影響）
                h = self.serotonin_system.modulate(h)
                
                # GABAによる調整（抑制性制御）
                h = self.gaba_system.modulate(h)
            
            # Layer 5 - 出力情報の生成
            elif i == 3:
                # ノルアドレナリンとグルタミン酸による調整（行動反応の強化）
                h = self.noradrenaline_system.modulate(h)
                h = self.glutamate_system.modulate(h)
            
            # レイヤーの順伝播
            h = layer(h)
        
        # 出力層（Layer 6）- ニューロモジュレーション出力
        output = self.layer6(h)
        
        return output
    
    def compute_loss(self, state: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        """
        損失の計算
        
        Args:
            state: 状態テンソル
            reward: 報酬テンソル
            
        Returns:
            計算された損失
        """
        # 状態の次元を調整
        if state.dim() == 1:
            state = state.unsqueeze(0)  # バッチ次元を追加
        
        # フォワードパスを実行
        output = self.forward(state)
        
        # 報酬に基づく目標値の計算
        target = torch.full_like(output, reward.item())
        
        # 損失関数（MSE）
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, target)
        
        # 報酬に基づく神経伝達物質レベルの調整
        reward_val = reward.item()
        
        # ドーパミンレベルの調整（報酬に基づく）
        self.dopamine_system.set_level(max(0.0, min(1.0, reward_val * 0.5 + 0.5)))
        
        # 高い報酬は興奮性伝達物質（グルタミン酸）レベルを上げる
        self.glutamate_system.set_level(max(0.0, min(1.0, reward_val * 0.3 + 0.5)))
        
        # 低い報酬は抑制性伝達物質（GABA）レベルを上げる
        if reward_val < 0:
            self.gaba_system.set_level(max(0.0, min(1.0, -reward_val * 0.3 + 0.5)))
        
        return loss
    
    def update(self, state: torch.Tensor, reward: torch.Tensor) -> float:
        """
        モデルの更新
        
        Args:
            state: 状態テンソル
            reward: 報酬テンソル
            
        Returns:
            計算された損失
        """
        # 勾配のリセット
        self.optimizer.zero_grad()
        
        # 損失の計算
        loss = self.compute_loss(state, reward)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        self.optimizer.step()
        
        return loss.item()
    
    def set_neurotransmitter_levels(
        self,
        dopamine: Optional[float] = None,
        serotonin: Optional[float] = None,
        acetylcholine: Optional[float] = None,
        noradrenaline: Optional[float] = None,
        gaba: Optional[float] = None,
        glutamate: Optional[float] = None
    ):
        """
        神経伝達物質レベルの設定
        
        Args:
            dopamine: ドーパミンレベル（0-1）
            serotonin: セロトニンレベル（0-1）
            acetylcholine: アセチルコリンレベル（0-1）
            noradrenaline: ノルアドレナリンレベル（0-1）
            gaba: GABAレベル（0-1）
            glutamate: グルタミン酸レベル（0-1）
        """
        if dopamine is not None:
            self.dopamine_system.set_level(dopamine)
            
        if serotonin is not None:
            self.serotonin_system.set_level(serotonin)
            
        if acetylcholine is not None:
            self.acetylcholine_system.set_level(acetylcholine)
            
        if noradrenaline is not None:
            self.noradrenaline_system.set_level(noradrenaline)
            
        if gaba is not None:
            self.gaba_system.set_level(gaba)
            
        if glutamate is not None:
            self.glutamate_system.set_level(glutamate)
            
    def get_neurotransmitter_levels(self) -> Dict[str, float]:
        """
        神経伝達物質レベルの取得
        
        Returns:
            各神経伝達物質のレベル
        """
        return {
            "dopamine": self.dopamine_system.level,
            "serotonin": self.serotonin_system.level,
            "acetylcholine": self.acetylcholine_system.level,
            "noradrenaline": self.noradrenaline_system.level,
            "gaba": self.gaba_system.level,
            "glutamate": self.glutamate_system.level
        }
        
    def get_neurotransmitter_effects(self) -> Dict[str, str]:
        """
        各神経伝達物質の効果の説明
        
        Returns:
            各神経伝達物質の効果説明
        """
        return {
            "dopamine": "報酬学習と動機付け - 高レベルは報酬への反応を強化します",
            "serotonin": "気分と感情の調整 - 高レベルは気分を安定させ、抑うつを軽減します",
            "acetylcholine": "注意力と記憶の処理 - 高レベルは注意力と学習能力を向上させます",
            "noradrenaline": "覚醒と注意の制御 - 高レベルは覚醒度を高め、環境への注意を促進します",
            "gaba": "神経活動の抑制 - 高レベルは過剰な神経活動を抑え、安定性を提供します",
            "glutamate": "神経活動の興奮 - 高レベルは情報処理を促進し、学習を強化します"
        }
        
    def simulate_drug_effect(self, drug_name: str, dose: float = 1.0) -> Dict[str, float]:
        """
        薬物効果のシミュレーション
        
        Args:
            drug_name: 薬物名
            dose: 投与量（0-1）
            
        Returns:
            変更された神経伝達物質レベル
        """
        # 薬物効果の定義
        drug_effects = {
            "methylphenidate": {  # リタリン
                "dopamine": 0.3,
                "noradrenaline": 0.4
            },
            "fluoxetine": {  # プロザック
                "serotonin": 0.5
            },
            "diazepam": {  # バリウム
                "gaba": 0.6
            },
            "donepezil": {  # アリセプト
                "acetylcholine": 0.5
            },
            "amphetamine": {  # アンフェタミン
                "dopamine": 0.7,
                "noradrenaline": 0.6
            },
            "ketamine": {  # ケタミン
                "glutamate": -0.5  # 抑制効果
            },
            "coffee": {  # カフェイン
                "noradrenaline": 0.3,
                "acetylcholine": 0.2
            },
            "alcohol": {  # アルコール
                "gaba": 0.4,
                "glutamate": -0.3
            }
        }
        
        if drug_name.lower() not in drug_effects:
            logger.warning(f"未知の薬物: {drug_name}")
            return self.get_neurotransmitter_levels()
        
        # 現在のレベルを取得
        current_levels = self.get_neurotransmitter_levels()
        
        # 薬物の効果を適用
        effects = drug_effects[drug_name.lower()]
        for transmitter, effect in effects.items():
            if transmitter in current_levels:
                # 用量を考慮した効果を計算
                adjusted_effect = effect * dose
                
                # 現在の値に効果を加算（0-1の範囲に制限）
                new_level = max(0.0, min(1.0, current_levels[transmitter] + adjusted_effect))
                
                # 神経伝達物質レベルを設定
                if transmitter == "dopamine":
                    self.dopamine_system.set_level(new_level)
                elif transmitter == "serotonin":
                    self.serotonin_system.set_level(new_level)
                elif transmitter == "acetylcholine":
                    self.acetylcholine_system.set_level(new_level)
                elif transmitter == "noradrenaline":
                    self.noradrenaline_system.set_level(new_level)
                elif transmitter == "gaba":
                    self.gaba_system.set_level(new_level)
                elif transmitter == "glutamate":
                    self.glutamate_system.set_level(new_level)
        
        return self.get_neurotransmitter_levels() 