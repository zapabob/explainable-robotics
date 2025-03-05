#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Explainable Roboticsデモスクリプト

このスクリプトは、神経伝達物質、LLM、Genesisロボットを統合したシステムのデモを実行します。
以下の機能をデモンストレーションします：
1. システムの初期化と設定
2. 神経伝達物質レベルの調整
3. 中枢神経系薬物の適用
4. ロボットの行動生成と説明
"""

import os
import time
import argparse
import logging
import json
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# モジュールのインポート
from explainable_robotics.core.integrated_system import create_integrated_system

def create_demo_config() -> Dict[str, Any]:
    """デモ用の設定ファイルを作成"""
    config = {
        'system': {
            'data_dir': './data',
            'knowledge_dir': './data/knowledge',
            'log_dir': './logs',
            'explanation_dir': './data/explanations'
        },
        'neurotransmitters': {
            'default_levels': {
                'acetylcholine': 0.5,
                'dopamine': 0.5,
                'serotonin': 0.5,
                'noradrenaline': 0.5,
                'glutamate': 0.5,
                'gaba': 0.5
            }
        },
        'cortical_model': {
            'layers': ['layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'],
            'input_dim': 100,
            'output_dim': 22
        },
        'robot_interface': {
            'simulation_mode': True,
            'connection': {
                'type': 'usb',
                'port': '/dev/ttyUSB0'
            },
            'sensors': {
                'camera': True,
                'imu': True,
                'joint_sensors': True,
                'force_sensors': True
            }
        },
        'llm': {
            'default_provider': 'openai',
            'openai': {
                'model': 'gpt-4o'
            },
            'anthropic': {
                'model': 'claude-3-opus-20240229'
            }
        }
    }
    
    # 設定ディレクトリの作成
    os.makedirs('./config', exist_ok=True)
    
    # 設定ファイルを保存
    with open('./config/demo_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    return config

def prepare_dummy_knowledge_base():
    """デモ用のダミー知識ベースを準備"""
    knowledge_dir = './data/knowledge'
    os.makedirs(knowledge_dir, exist_ok=True)
    
    # 神経科学の知識ファイル
    neuroscience_content = """
    # 神経伝達物質と脳機能
    
    ## アセチルコリン (ACh)
    - 記憶形成と注意に関与
    - アルツハイマー病では減少
    - ムスカリン受容体とニコチン受容体に作用
    
    ## ドーパミン (DA)
    - 報酬系とモチベーションに関与
    - パーキンソン病では減少、統合失調症では過剰
    - D1からD5までの受容体サブタイプがある
    
    ## セロトニン (5-HT)
    - 気分、睡眠、食欲の調節
    - うつ病では機能不全
    - 多数の受容体サブタイプ（5-HT1A, 5-HT2A等）
    
    ## ノルアドレナリン (NA)
    - 覚醒と注意、ストレス反応に関与
    - アドレナリンの前駆体
    - α受容体とβ受容体に作用
    
    ## グルタミン酸
    - 主要な興奮性神経伝達物質
    - 学習と記憶に重要
    - NMDA、AMPA、カイニン酸、代謝型受容体に作用
    
    ## GABA
    - 主要な抑制性神経伝達物質
    - 不安の調整に重要
    - GABA-A（イオンチャネル型）とGABA-B（Gタンパク質共役型）受容体
    """
    
    with open(os.path.join(knowledge_dir, 'neuroscience.md'), 'w', encoding='utf-8') as f:
        f.write(neuroscience_content)
    
    # 薬理学の知識ファイル
    pharmacology_content = """
    # 中枢神経系薬物と作用機序
    
    ## 精神刺激薬
    - メチルフェニデート (Ritalin): ドーパミン再取り込み阻害
    - アンフェタミン: ドーパミン放出増加と再取り込み阻害
    - カフェイン: アデノシン受容体阻害、覚醒効果
    
    ## 抗不安薬
    - ジアゼパム (Valium): GABA-A受容体の作用増強
    - ゾルピデム (Ambien): GABA-A受容体のベンゾジアゼピン部位に特異的に結合
    
    ## 抗うつ薬
    - フルオキセチン (Prozac): 選択的セロトニン再取り込み阻害薬 (SSRI)
    - ベンラファキシン (Effexor): セロトニン・ノルアドレナリン再取り込み阻害薬 (SNRI)
    
    ## 抗精神病薬
    - ハロペリドール (Haldol): D2ドーパミン受容体遮断薬
    - クロザピン (Clozaril): 非定型抗精神病薬、多数の受容体に作用
    
    ## 認知症治療薬
    - ドネペジル (Aricept): アセチルコリンエステラーゼ阻害薬
    
    ## アルコール
    - GABA-A受容体作用増強
    - NMDA受容体機能抑制
    """
    
    with open(os.path.join(knowledge_dir, 'pharmacology.md'), 'w', encoding='utf-8') as f:
        f.write(pharmacology_content)

def visualize_neurotransmitter_levels(nt_levels: Dict[str, float], title: str = "神経伝達物質レベル"):
    """神経伝達物質レベルを可視化"""
    plt.figure(figsize=(10, 6))
    names = list(nt_levels.keys())
    values = list(nt_levels.values())
    
    bars = plt.bar(names, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'])
    
    # 値をバーの上に表示
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f'{value:.2f}',
            ha='center',
            va='bottom'
        )
    
    plt.ylim(0, 1.1)
    plt.title(title)
    plt.ylabel("レベル (0-1)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存するディレクトリが存在しない場合は作成
    os.makedirs('./outputs', exist_ok=True)
    
    # タイムスタンプを含めてファイル名を生成
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f'./outputs/nt_levels_{timestamp}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"神経伝達物質レベルのグラフを保存しました: {filename}")
    plt.close()

def demo_baseline(system):
    """ベースラインのシステム状態をデモ"""
    logger.info("=== ベースラインデモ ===")
    
    # システムを開始
    system.start()
    time.sleep(2)  # 安定化のために少し待機
    
    # 現在の神経伝達物質レベルを取得して表示
    state = system.explain_current_state()
    nt_levels = state['neurotransmitter_levels']
    
    logger.info("現在の神経伝達物質レベル:")
    for nt, level in nt_levels.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels, "ベースライン神経伝達物質レベル")
    
    # 自然言語の説明を生成
    explanation = system.get_natural_language_explanation(
        "現在のシステム状態と神経伝達物質レベルが、ロボットの行動にどのように影響していますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ジェスチャーを実行
    system.execute_gesture("wave")
    time.sleep(3)
    
    return nt_levels

def demo_dopamine_modulation(system, baseline_levels):
    """ドーパミンレベルの調整デモ"""
    logger.info("=== ドーパミン調整デモ ===")
    
    # ドーパミンレベルを上げる
    high_dopamine = 0.8
    system.adjust_neurotransmitter('dopamine', high_dopamine)
    time.sleep(2)  # 効果が現れるまで待機
    
    # 状態を確認
    state_high = system.explain_current_state()
    nt_levels_high = state_high['neurotransmitter_levels']
    
    logger.info("ドーパミン上昇後の神経伝達物質レベル:")
    for nt, level in nt_levels_high.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_high, "ドーパミン上昇後の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "ドーパミンレベルの上昇は、ロボットの行動にどのような影響を与えていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ジェスチャーを実行（ドーパミン上昇でより活発に）
    system.execute_gesture("dance")
    time.sleep(3)
    
    # ドーパミンレベルを下げる
    low_dopamine = 0.2
    system.adjust_neurotransmitter('dopamine', low_dopamine)
    time.sleep(2)
    
    # 状態を確認
    state_low = system.explain_current_state()
    nt_levels_low = state_low['neurotransmitter_levels']
    
    logger.info("ドーパミン低下後の神経伝達物質レベル:")
    for nt, level in nt_levels_low.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_low, "ドーパミン低下後の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "ドーパミンレベルの低下は、ロボットの行動にどのような影響を与えていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ベースラインに戻す
    for nt, level in baseline_levels.items():
        system.adjust_neurotransmitter(nt, level)
    time.sleep(2)

def demo_glutamate_gaba_balance(system, baseline_levels):
    """グルタミン酸とGABAのバランスデモ"""
    logger.info("=== グルタミン酸・GABAバランスデモ ===")
    
    # グルタミン酸を上げ、GABAを下げる（興奮優位）
    system.adjust_neurotransmitter('glutamate', 0.8)
    system.adjust_neurotransmitter('gaba', 0.2)
    time.sleep(2)
    
    # 状態を確認
    state_excitatory = system.explain_current_state()
    nt_levels_excitatory = state_excitatory['neurotransmitter_levels']
    
    logger.info("興奮優位（グルタミン酸↑、GABA↓）後の神経伝達物質レベル:")
    for nt, level in nt_levels_excitatory.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_excitatory, "興奮優位の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "グルタミン酸の増加とGABAの減少（興奮優位）は、ロボットの皮質活動にどのような影響を与えていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ジェスチャーを実行
    system.execute_gesture("excited_movement")
    time.sleep(3)
    
    # グルタミン酸を下げ、GABAを上げる（抑制優位）
    system.adjust_neurotransmitter('glutamate', 0.2)
    system.adjust_neurotransmitter('gaba', 0.8)
    time.sleep(2)
    
    # 状態を確認
    state_inhibitory = system.explain_current_state()
    nt_levels_inhibitory = state_inhibitory['neurotransmitter_levels']
    
    logger.info("抑制優位（グルタミン酸↓、GABA↑）後の神経伝達物質レベル:")
    for nt, level in nt_levels_inhibitory.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_inhibitory, "抑制優位の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "グルタミン酸の減少とGABAの増加（抑制優位）は、ロボットの皮質活動にどのような影響を与えていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ジェスチャーを実行
    system.execute_gesture("slow_movement")
    time.sleep(3)
    
    # グルタミン酸受容体の感受性調整
    logger.info("グルタミン酸受容体感受性の調整")
    receptor_changes = {
        'nmda': 1.5,   # NMDA受容体の感受性を上げる
        'ampa': 0.8    # AMPA受容体の感受性を下げる
    }
    system.adjust_receptor_sensitivity('glutamate', receptor_changes)
    time.sleep(2)
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "NMDA受容体の感受性増加とAMPA受容体の感受性低下は、ロボットの学習能力にどのような影響を与えますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # ベースラインに戻す
    for nt, level in baseline_levels.items():
        system.adjust_neurotransmitter(nt, level)
    time.sleep(2)

def demo_drug_effects(system, baseline_levels):
    """中枢神経系薬物の効果デモ"""
    logger.info("=== 薬物効果デモ ===")
    
    # メチルフェニデート（リタリン）の効果
    logger.info("メチルフェニデートを適用")
    system.apply_drug('methylphenidate', 1.0)
    time.sleep(2)
    
    # 状態を確認
    state_methylphenidate = system.explain_current_state()
    nt_levels_methylphenidate = state_methylphenidate['neurotransmitter_levels']
    drug_effects = state_methylphenidate['drug_effects']
    
    logger.info("メチルフェニデート投与後の神経伝達物質レベル:")
    for nt, level in nt_levels_methylphenidate.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    logger.info("薬物効果:")
    for drug, effect in drug_effects.items():
        logger.info(f"  - {drug}: {effect}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_methylphenidate, "メチルフェニデート投与後の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "メチルフェニデート（リタリン）の投与は、ロボットの注意力と行動にどのように影響していますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # 薬物をクリア
    system.drug_system.clear_drug_effects()
    time.sleep(1)
    
    # ジアゼパム（バリウム）の効果
    logger.info("ジアゼパムを適用")
    system.apply_drug('diazepam', 1.0)
    time.sleep(2)
    
    # 状態を確認
    state_diazepam = system.explain_current_state()
    nt_levels_diazepam = state_diazepam['neurotransmitter_levels']
    drug_effects = state_diazepam['drug_effects']
    
    logger.info("ジアゼパム投与後の神経伝達物質レベル:")
    for nt, level in nt_levels_diazepam.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_diazepam, "ジアゼパム投与後の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "ジアゼパム（バリウム）の投与は、ロボットの行動にどのような抑制効果をもたらしていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # 薬物をクリア
    system.drug_system.clear_drug_effects()
    time.sleep(1)
    
    # ドネペジル（アリセプト）の効果
    logger.info("ドネペジルを適用")
    system.apply_drug('donepezil', 1.0)
    time.sleep(2)
    
    # 状態を確認
    state_donepezil = system.explain_current_state()
    nt_levels_donepezil = state_donepezil['neurotransmitter_levels']
    
    logger.info("ドネペジル投与後の神経伝達物質レベル:")
    for nt, level in nt_levels_donepezil.items():
        logger.info(f"  - {nt}: {level:.2f}")
    
    # 可視化
    visualize_neurotransmitter_levels(nt_levels_donepezil, "ドネペジル投与後の神経伝達物質レベル")
    
    # 行動の説明を生成
    explanation = system.get_natural_language_explanation(
        "ドネペジル（アリセプト）の投与は、ロボットの認知機能と記憶にどのような影響を与えていますか？"
    )
    logger.info(f"説明: {explanation}")
    
    # 薬物をクリアしてベースラインに戻す
    system.drug_system.clear_drug_effects()
    for nt, level in baseline_levels.items():
        system.adjust_neurotransmitter(nt, level)
    time.sleep(2)

def run_demo():
    """デモを実行します"""
    logger.info("ExplainableRoboticsデモを開始します")
    
    # デモ用の設定を作成
    config = create_demo_config()
    
    # 知識ベースを準備
    prepare_dummy_knowledge_base()
    
    # システムを初期化
    system = create_integrated_system('./config/demo_config.json')
    
    try:
        # 知識ベースを初期化
        system.initialize_knowledge_base()
        
        # ベースラインデモ
        baseline_levels = demo_baseline(system)
        
        # ドーパミン調整デモ
        demo_dopamine_modulation(system, baseline_levels)
        
        # グルタミン酸とGABAのバランスデモ
        demo_glutamate_gaba_balance(system, baseline_levels)
        
        # 薬物効果デモ
        demo_drug_effects(system, baseline_levels)
        
        # システムを停止
        system.stop()
        
        logger.info("デモが正常に完了しました")
        
    except Exception as e:
        logger.error(f"デモの実行中にエラーが発生しました: {str(e)}")
        system.stop()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Explainable Roboticsデモ')
    args = parser.parse_args()
    
    run_demo() 