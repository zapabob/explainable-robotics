import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class NeurotransmitterSystem:
    """
    神経伝達物質システムの基本クラス。
    様々な神経伝達物質レベルを管理し、ニューラルネットワークの挙動に影響を与えます。
    """
    
    def __init__(self):
        # 基本的な神経伝達物質レベルの初期化
        self.levels = {
            'acetylcholine': 0.5,  # 注意と記憶
            'dopamine': 0.5,       # 報酬と動機
            'serotonin': 0.5,      # 気分と感情
            'noradrenaline': 0.5,  # 覚醒と注意
            'glutamate': 0.5,      # 興奮性伝達物質
            'gaba': 0.5,           # 抑制性伝達物質
        }
        
        # 薬物影響のマッピングを初期化
        self.drug_effects = {}
        self._init_drug_effects()
        
        # 現在活性化されている薬物
        self.active_drugs = []
        
    def _init_drug_effects(self):
        """
        様々な中枢神経作用薬の効果を初期化します。
        各薬物は複数の神経伝達物質レベルに影響を与える可能性があります。
        """
        # 刺激薬
        self.drug_effects['methylphenidate'] = {  # リタリン
            'dopamine': +0.3,
            'noradrenaline': +0.2
        }
        self.drug_effects['amphetamine'] = {  # アデラール
            'dopamine': +0.4,
            'noradrenaline': +0.3,
            'serotonin': +0.1
        }
        
        # 抗不安薬・睡眠薬
        self.drug_effects['diazepam'] = {  # バリウム
            'gaba': +0.4,
            'glutamate': -0.2
        }
        self.drug_effects['zolpidem'] = {  # アンビエン
            'gaba': +0.5
        }
        
        # 抗うつ薬
        self.drug_effects['fluoxetine'] = {  # プロザック
            'serotonin': +0.4
        }
        self.drug_effects['venlafaxine'] = {  # エフェクサー
            'serotonin': +0.3,
            'noradrenaline': +0.2
        }
        
        # 抗精神病薬
        self.drug_effects['haloperidol'] = {  # ハルドール
            'dopamine': -0.5
        }
        self.drug_effects['clozapine'] = {  # クロザリル
            'dopamine': -0.3,
            'serotonin': -0.2,
            'acetylcholine': -0.2
        }
        
        # 認知症治療薬
        self.drug_effects['donepezil'] = {  # アリセプト
            'acetylcholine': +0.4
        }
        
        # アルコール
        self.drug_effects['alcohol'] = {
            'gaba': +0.3,
            'glutamate': -0.3,
            'dopamine': +0.2
        }
        
        # カフェイン
        self.drug_effects['caffeine'] = {
            'acetylcholine': +0.2,
            'noradrenaline': +0.2
        }
    
    def set_level(self, transmitter_type: str, level: float, target_regions: Optional[List[str]] = None) -> None:
        """
        特定の神経伝達物質のレベルを設定します。
        
        Args:
            transmitter_type: 神経伝達物質の種類
            level: 設定するレベル（0.0〜1.0）
            target_regions: 影響を与える特定の脳領域（オプション）
        """
        if transmitter_type not in self.levels:
            logger.warning(f"未知の神経伝達物質: {transmitter_type}")
            return
            
        level = max(0.0, min(1.0, level))  # 0.0〜1.0の範囲に制限
        self.levels[transmitter_type] = level
        logger.info(f"{transmitter_type}のレベルを{level:.2f}に設定しました")
    
    def get_level(self, transmitter_type: str) -> float:
        """
        特定の神経伝達物質の現在のレベルを取得します。
        
        Args:
            transmitter_type: 神経伝達物質の種類
            
        Returns:
            現在のレベル値（0.0〜1.0）
        """
        if transmitter_type not in self.levels:
            logger.warning(f"未知の神経伝達物質: {transmitter_type}")
            return 0.5  # デフォルト値を返す
            
        return self.levels[transmitter_type]

    def apply_drug(self, drug_name: str, strength: float = 1.0) -> None:
        """
        薬物の効果を神経伝達物質レベルに適用します。
        
        Args:
            drug_name: 薬物の名前
            strength: 効果の強さ（0.0〜1.0）
        """
        if drug_name not in self.drug_effects:
            logger.warning(f"未知の薬物: {drug_name}")
            return
            
        if drug_name not in self.active_drugs:
            self.active_drugs.append(drug_name)
            
        effects = self.drug_effects[drug_name]
        for transmitter, effect in effects.items():
            current = self.levels[transmitter]
            new_level = max(0.0, min(1.0, current + effect * strength))
            self.levels[transmitter] = new_level
            
        logger.info(f"{drug_name}を適用しました（強度: {strength:.2f}）")
        
    def clear_drug(self, drug_name: str) -> None:
        """
        特定の薬物の効果をクリアします。
        
        Args:
            drug_name: 薬物の名前
        """
        if drug_name not in self.active_drugs:
            return
            
        self.active_drugs.remove(drug_name)
        # この実装は単純化されています。実際には徐々に効果が消えるなど、より複雑な処理が必要です。
        logger.info(f"{drug_name}の効果をクリアしました")
        
    def clear_all_drugs(self) -> None:
        """全ての薬物効果をクリアして初期状態に戻します。"""
        self.active_drugs = []
        # 全てのレベルをデフォルトに戻す
        for transmitter in self.levels:
            self.levels[transmitter] = 0.5
        logger.info("全ての薬物効果をクリアしました")
    
    def get_modulation_factors(self) -> Dict[str, float]:
        """
        現在の神経伝達物質レベルに基づく変調係数を取得します。
        これらの係数はニューラルネットワークの挙動に影響を与えます。
        
        Returns:
            変調係数の辞書
        """
        factors = {}
        
        # アセチルコリン - 注意と記憶の変調
        # 高レベル：入力感度↑、シナプス可塑性↑
        factors['input_sensitivity'] = 0.5 + 0.5 * self.levels['acetylcholine']
        factors['synaptic_plasticity'] = 0.5 + 0.5 * self.levels['acetylcholine']
        
        # ドーパミン - 報酬と動機の変調
        # 高レベル：探索行動↑、行動選択の変動性↑
        factors['exploration'] = 0.5 + 0.5 * self.levels['dopamine']
        factors['action_variability'] = 0.2 + 0.8 * self.levels['dopamine']
        
        # セロトニン - 情緒と衝動の変調
        # 高レベル：リスク回避↑、長期的報酬の重視↑
        factors['risk_aversion'] = 0.5 + 0.5 * self.levels['serotonin']
        factors['patience'] = 0.2 + 0.8 * self.levels['serotonin']
        
        # ノルアドレナリン - 覚醒と注意の変調
        # 高レベル：反応速度↑、外部刺激への感度↑
        factors['response_speed'] = 0.2 + 0.8 * self.levels['noradrenaline']
        factors['external_focus'] = 0.2 + 0.8 * self.levels['noradrenaline']
        
        # グルタミン酸 - 興奮性伝達物質
        # 高レベル：学習速度↑、活性化強度↑、可塑性↑
        factors['learning_rate'] = 0.2 + 0.8 * self.levels['glutamate']
        factors['activation_strength'] = 0.5 + 0.5 * self.levels['glutamate']
        
        # GABA - 抑制性伝達物質
        # 高レベル：ノイズ抑制↑、発火閾値↑、緊張の低下↑
        factors['noise_suppression'] = 0.2 + 0.8 * self.levels['gaba']
        factors['firing_threshold'] = 0.2 + 0.8 * self.levels['gaba']
        
        # グルタミン酸とGABAのバランス - E/I比に影響
        # このバランスが崩れると様々な神経疾患の状態を模倣できます
        e_i_ratio = self.levels['glutamate'] / max(0.1, self.levels['gaba'])
        factors['e_i_balance'] = min(2.0, e_i_ratio)  # 比率の上限を設定
        
        return factors
        
    def get_active_drugs(self) -> List[str]:
        """現在活性化されている薬物のリストを返します。"""
        return self.active_drugs.copy()
        
    def describe_current_state(self) -> Dict[str, Any]:
        """
        現在の神経伝達物質の状態と、それが行動にどのように影響するかを説明します。
        
        Returns:
            状態の説明を含む辞書
        """
        description = {
            'neurotransmitter_levels': self.levels.copy(),
            'active_drugs': self.active_drugs.copy(),
            'behavioral_effects': {}
        }
        
        # アセチルコリンの効果を説明
        ach_level = self.levels['acetylcholine']
        if ach_level > 0.7:
            description['behavioral_effects']['acetylcholine'] = "注意力が高く、新しい情報の処理と記憶が強化されています。"
        elif ach_level < 0.3:
            description['behavioral_effects']['acetylcholine'] = "注意力が低下し、記憶形成が困難になっています。"
        else:
            description['behavioral_effects']['acetylcholine'] = "通常の注意力と記憶力を維持しています。"
            
        # ドーパミンの効果を説明
        dop_level = self.levels['dopamine']
        if dop_level > 0.7:
            description['behavioral_effects']['dopamine'] = "動機付けが高く、報酬を積極的に追求します。探索行動が増加しています。"
        elif dop_level < 0.3:
            description['behavioral_effects']['dopamine'] = "動機付けが低く、無感動状態です。新たな行動を起こす意欲が減少しています。"
        else:
            description['behavioral_effects']['dopamine'] = "適度な動機付けと報酬感受性を維持しています。"
            
        # セロトニンの効果を説明
        ser_level = self.levels['serotonin']
        if ser_level > 0.7:
            description['behavioral_effects']['serotonin'] = "感情状態が安定し、長期的視点で意思決定を行います。衝動性が低下しています。"
        elif ser_level < 0.3:
            description['behavioral_effects']['serotonin'] = "感情の不安定さと衝動性が増加しています。短期的な報酬を優先する傾向があります。"
        else:
            description['behavioral_effects']['serotonin'] = "通常の感情バランスと衝動制御を維持しています。"
            
        # ノルアドレナリンの効果を説明
        nor_level = self.levels['noradrenaline']
        if nor_level > 0.7:
            description['behavioral_effects']['noradrenaline'] = "警戒状態が高く、外部刺激への反応が素早くなっています。"
        elif nor_level < 0.3:
            description['behavioral_effects']['noradrenaline'] = "警戒心が低く、環境変化への反応が遅くなっています。"
        else:
            description['behavioral_effects']['noradrenaline'] = "適正な警戒レベルを維持しています。"
            
        # グルタミン酸の効果を説明
        glu_level = self.levels['glutamate']
        if glu_level > 0.7:
            description['behavioral_effects']['glutamate'] = "神経活動が活発で、学習速度が上がっていますが、過剰興奮状態となっています。"
        elif glu_level < 0.3:
            description['behavioral_effects']['glutamate'] = "神経活動が低下し、学習能力や反応速度が低下しています。"
        else:
            description['behavioral_effects']['glutamate'] = "適切な神経興奮レベルを維持しています。"
            
        # GABAの効果を説明
        gaba_level = self.levels['gaba']
        if gaba_level > 0.7:
            description['behavioral_effects']['gaba'] = "神経活動が抑制され、落ち着いた状態ですが、認知処理速度が低下しています。"
        elif gaba_level < 0.3:
            description['behavioral_effects']['gaba'] = "神経抑制が弱く、過敏に反応する可能性があります。不安感が増大している可能性があります。"
        else:
            description['behavioral_effects']['gaba'] = "適切な神経抑制レベルを維持しています。"
            
        # E/I比の効果を説明
        factors = self.get_modulation_factors()
        e_i_balance = factors['e_i_balance']
        if e_i_balance > 1.5:
            description['behavioral_effects']['e_i_balance'] = "興奮/抑制バランスが興奮側に傾いています。行動が活発ですが、制御が難しい可能性があります。"
        elif e_i_balance < 0.7:
            description['behavioral_effects']['e_i_balance'] = "興奮/抑制バランスが抑制側に傾いています。行動が抑制され、反応性が低下しています。"
        else:
            description['behavioral_effects']['e_i_balance'] = "興奮/抑制のバランスが取れています。"
            
        return description


class GlutamateSystem:
    """
    グルタミン酸系の神経伝達を制御するシステム。
    グルタミン酸は主要な興奮性神経伝達物質で、学習と記憶に重要です。
    """
    
    def __init__(self):
        self.base_level = 0.5
        self.current_level = 0.5
        # グルタミン酸受容体の状態
        self.receptors = {
            'AMPA': 1.0,  # 速い興奮性シナプス伝達
            'NMDA': 1.0,  # 学習と記憶に関与
            'kainate': 1.0,  # 興奮性シナプス伝達の調節
            'mGluR': 1.0   # 代謝型グルタミン酸受容体
        }
    
    def modulate(self, level_change: float, target_receptors: Optional[Dict[str, float]] = None) -> None:
        """
        グルタミン酸レベルとその受容体感受性を調整します。
        
        Args:
            level_change: グルタミン酸レベルの変化量
            target_receptors: 特定の受容体に対する変化（辞書形式）
        """
        # 全体レベルの調整
        self.current_level = max(0.0, min(1.0, self.current_level + level_change))
        
        # 特定の受容体調整
        if target_receptors:
            for receptor, change in target_receptors.items():
                if receptor in self.receptors:
                    self.receptors[receptor] = max(0.1, min(2.0, self.receptors[receptor] + change))
    
    def get_effect_on_layer(self, layer_type: str) -> Dict[str, float]:
        """
        特定の皮質層に対するグルタミン酸の効果を計算します。
        
        Args:
            layer_type: 皮質層のタイプ
            
        Returns:
            効果パラメータを含む辞書
        """
        effects = {}
        
        # グルタミン酸は基本的に全ての層に興奮性効果をもたらす
        effects['excitation'] = self.current_level * self.receptors['AMPA']
        
        # 特に層2/3と層5では、NMDAを介した可塑性に影響
        if layer_type in ['layer2_3', 'layer5']:
            effects['plasticity'] = self.current_level * self.receptors['NMDA']
        
        # 層4では感覚入力処理に影響
        if layer_type == 'layer4':
            effects['sensory_gain'] = self.current_level * self.receptors['kainate']
        
        # 代謝型受容体は長期的な変化に影響
        effects['adaptation'] = self.current_level * self.receptors['mGluR']
        
        return effects


class GABASystem:
    """
    GABA系の神経伝達を制御するシステム。
    GABAは主要な抑制性神経伝達物質で、神経ネットワークの安定性に重要です。
    """
    
    def __init__(self):
        self.base_level = 0.5
        self.current_level = 0.5
        # GABA受容体の状態
        self.receptors = {
            'GABA_A': 1.0,  # 速い抑制性シナプス伝達
            'GABA_B': 1.0   # 遅い抑制性シナプス伝達
        }
    
    def modulate(self, level_change: float, target_receptors: Optional[Dict[str, float]] = None) -> None:
        """
        GABAレベルとその受容体感受性を調整します。
        
        Args:
            level_change: GABAレベルの変化量
            target_receptors: 特定の受容体に対する変化（辞書形式）
        """
        # 全体レベルの調整
        self.current_level = max(0.0, min(1.0, self.current_level + level_change))
        
        # 特定の受容体調整
        if target_receptors:
            for receptor, change in target_receptors.items():
                if receptor in self.receptors:
                    self.receptors[receptor] = max(0.1, min(2.0, self.receptors[receptor] + change))
    
    def get_effect_on_layer(self, layer_type: str) -> Dict[str, float]:
        """
        特定の皮質層に対するGABAの効果を計算します。
        
        Args:
            layer_type: 皮質層のタイプ
            
        Returns:
            効果パラメータを含む辞書
        """
        effects = {}
        
        # GABAは基本的に全ての層に抑制性効果をもたらす
        effects['inhibition'] = self.current_level * self.receptors['GABA_A']
        
        # 特に層2/3では側方抑制に影響
        if layer_type == 'layer2_3':
            effects['lateral_inhibition'] = self.current_level * self.receptors['GABA_A']
        
        # 層4と層5ではGABA_Bを介した長期抑制に影響
        if layer_type in ['layer4', 'layer5']:
            effects['long_term_inhibition'] = self.current_level * self.receptors['GABA_B']
        
        # ノイズ抑制効果
        effects['noise_reduction'] = self.current_level * self.receptors['GABA_A']
        
        return effects


class DrugSystem:
    """
    薬物とその神経伝達物質への影響を管理するシステム。
    様々な医薬品や薬物の効果をシミュレートします。
    """
    
    def __init__(self, neurotransmitter_system: NeurotransmitterSystem):
        self.neurotransmitter_system = neurotransmitter_system
        # 薬物の代謝時間（シミュレーション時間単位）
        self.metabolism_rates = {
            'methylphenidate': 0.05,  # リタリン
            'amphetamine': 0.03,      # アデラール
            'diazepam': 0.02,         # バリウム
            'zolpidem': 0.1,          # アンビエン
            'fluoxetine': 0.01,       # プロザック
            'venlafaxine': 0.02,      # エフェクサー
            'haloperidol': 0.02,      # ハルドール
            'clozapine': 0.03,        # クロザリル
            'donepezil': 0.01,        # アリセプト
            'alcohol': 0.08,
            'caffeine': 0.05
        }
        
        # 現在の薬物レベル
        self.current_levels = {}
        
    def apply_drug(self, drug_name: str, dose: float = 1.0) -> None:
        """
        薬物を適用し、その効果を神経伝達物質システムに反映します。
        
        Args:
            drug_name: 薬物名
            dose: 投与量（相対値）
        """
        if drug_name not in self.neurotransmitter_system.drug_effects:
            logger.warning(f"未知の薬物: {drug_name}")
            return
            
        # 薬物レベルを追加（累積可能）
        if drug_name in self.current_levels:
            self.current_levels[drug_name] = min(2.0, self.current_levels[drug_name] + dose)
        else:
            self.current_levels[drug_name] = dose
            
        # 神経伝達物質システムに薬物効果を適用
        self.neurotransmitter_system.apply_drug(drug_name, self.current_levels[drug_name])
        logger.info(f"{drug_name}を適用しました（用量: {dose:.2f}）")
            
    def metabolism_step(self) -> None:
        """
        薬物代謝のシミュレーションステップを実行します。
        時間経過とともに薬物レベルが低下します。
        """
        drugs_to_remove = []
        
        for drug, level in self.current_levels.items():
            if drug in self.metabolism_rates:
                # 代謝による薬物レベルの低下
                new_level = level - self.metabolism_rates[drug]
                if new_level <= 0:
                    drugs_to_remove.append(drug)
                    self.neurotransmitter_system.clear_drug(drug)
                else:
                    self.current_levels[drug] = new_level
                    # 薬物効果を更新
                    self.neurotransmitter_system.clear_drug(drug)
                    self.neurotransmitter_system.apply_drug(drug, new_level)
                    
        # 完全に代謝された薬物を削除
        for drug in drugs_to_remove:
            del self.current_levels[drug]
            
    def get_current_drug_levels(self) -> Dict[str, float]:
        """現在の薬物レベルを取得します。"""
        return self.current_levels.copy()
        
    def describe_drug_effects(self) -> Dict[str, str]:
        """
        現在活性化されている薬物とその効果の説明を提供します。
        
        Returns:
            薬物名とその効果の説明を含む辞書
        """
        descriptions = {}
        
        for drug, level in self.current_levels.items():
            intensity = "強い" if level > 1.5 else "中程度の" if level > 0.8 else "弱い"
            
            if drug == 'methylphenidate':
                descriptions[drug] = f"{intensity}集中力向上効果。ドーパミンとノルアドレナリンレベルが上昇しています。"
            elif drug == 'amphetamine':
                descriptions[drug] = f"{intensity}覚醒と興奮効果。ドーパミン、ノルアドレナリン、セロトニンレベルが上昇しています。"
            elif drug == 'diazepam':
                descriptions[drug] = f"{intensity}抗不安・鎮静効果。GABA活性が増加し、グルタミン酸活性が減少しています。"
            elif drug == 'zolpidem':
                descriptions[drug] = f"{intensity}睡眠誘発効果。GABA活性が大幅に増加しています。"
            elif drug == 'fluoxetine':
                descriptions[drug] = f"{intensity}抗うつ効果。セロトニンレベルが上昇しています。"
            elif drug == 'venlafaxine':
                descriptions[drug] = f"{intensity}抗うつと抗不安効果。セロトニンとノルアドレナリンレベルが上昇しています。"
            elif drug == 'haloperidol':
                descriptions[drug] = f"{intensity}抗精神病効果。ドーパミンレベルが大幅に低下しています。"
            elif drug == 'clozapine':
                descriptions[drug] = f"{intensity}抗精神病効果。ドーパミン、セロトニン、アセチルコリンレベルが低下しています。"
            elif drug == 'donepezil':
                descriptions[drug] = f"{intensity}認知機能向上効果。アセチルコリンレベルが上昇しています。"
            elif drug == 'alcohol':
                descriptions[drug] = f"{intensity}抑制と覚醒低下効果。GABA活性が増加し、グルタミン酸活性が減少しています。"
            elif drug == 'caffeine':
                descriptions[drug] = f"{intensity}覚醒効果。アセチルコリンとノルアドレナリンレベルが上昇しています。"
            else:
                descriptions[drug] = f"{intensity}効果が現れています。"
                
        return descriptions 