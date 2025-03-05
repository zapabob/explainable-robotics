#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ExplainableRobotics メインプログラム - コマンドラインインターフェース
"""

import os
import sys
import json
import time
import logging
import argparse
import cmd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from explainable_robotics.core.integrated_system import create_integrated_system

class ExplainableRoboticsShell(cmd.Cmd):
    """ExplainableRoboticsのインタラクティブシェル"""
    
    intro = "ExplainableRobotics インタラクティブシェルへようこそ。ヘルプが必要な場合は help または ? を入力してください。"
    prompt = "explainable_robotics> "
    
    def __init__(self, config_path=None):
        super().__init__()
        self.config_path = config_path
        self.system = None
        self.logger = logging.getLogger("explainable_robotics.cli")
    
    def preloop(self):
        """シェル起動前の処理"""
        self.logger.info("ExplainableRoboticsシェルを起動します")
        
        if self.config_path and os.path.exists(self.config_path):
            self.logger.info(f"設定ファイル {self.config_path} を読み込みます")
            try:
                self.system = create_integrated_system(self.config_path)
                self.logger.info("システムを初期化しました")
                print(f"システムを初期化しました（設定: {self.config_path}）")
            except Exception as e:
                self.logger.error(f"設定ファイルの読み込みエラー: {str(e)}")
                print(f"エラー: 設定ファイルの読み込みに失敗しました - {str(e)}")
        else:
            self.logger.warning("設定ファイルが指定されていないか、存在しません")
            print("注意: 設定ファイルが指定されていません。'start'コマンドで設定ファイルを指定してください。")
    
    def postloop(self):
        """シェル終了時の処理"""
        if self.system:
            try:
                self.system.stop()
                self.logger.info("システムを停止しました")
                print("システムを停止しました")
            except Exception as e:
                self.logger.error(f"システム停止エラー: {str(e)}")
                print(f"エラー: システム停止中にエラーが発生しました - {str(e)}")
    
    def do_start(self, arg):
        """システムを開始します。オプションとして設定ファイルを指定できます: start [config_path]"""
        args = arg.strip().split()
        config_path = args[0] if args else self.config_path
        
        if not config_path:
            print("エラー: 設定ファイルが指定されていません")
            self.logger.error("設定ファイルが指定されていません")
            return
        
        if not os.path.exists(config_path):
            print(f"エラー: 設定ファイル '{config_path}' が見つかりません")
            self.logger.error(f"設定ファイル '{config_path}' が見つかりません")
            return
        
        try:
            self.system = create_integrated_system(config_path)
            self.logger.info(f"設定ファイル '{config_path}' でシステムを初期化しました")
            
            print("知識ベースを初期化しています...")
            self.system.initialize_knowledge_base()
            self.logger.info("知識ベースを初期化しました")
            
            print("システムを開始します...")
            self.system.start()
            self.logger.info("システムを開始しました")
            print("システムが正常に開始されました")
        except Exception as e:
            self.logger.error(f"システム開始エラー: {str(e)}")
            print(f"エラー: システムの開始に失敗しました - {str(e)}")
    
    def do_stop(self, arg):
        """システムを停止します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            self.system.stop()
            self.logger.info("システムを停止しました")
            print("システムが正常に停止されました")
        except Exception as e:
            self.logger.error(f"システム停止エラー: {str(e)}")
            print(f"エラー: システムの停止に失敗しました - {str(e)}")
    
    def do_state(self, arg):
        """システムの現在の状態を表示します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            state = self.system.explain_current_state()
            print("\n===== システム状態 =====")
            print(f"実行状態: {'実行中' if self.system.is_running else '停止中'}")
            
            print("\n--- 神経伝達物質レベル ---")
            for nt, level in state['neurotransmitter_levels'].items():
                print(f"{nt}: {level:.2f}")
            
            print("\n--- アクティブな薬物効果 ---")
            if 'active_drugs' in state and state['active_drugs']:
                for drug, info in state['active_drugs'].items():
                    print(f"{drug} (強度: {info['dose']:.2f}, 残り時間: {info['time_left']:.1f}秒)")
            else:
                print("アクティブな薬物はありません")
            
            if 'behavioral_effects' in state:
                print("\n--- 行動への影響 ---")
                print(state['behavioral_effects'])
        except Exception as e:
            self.logger.error(f"状態取得エラー: {str(e)}")
            print(f"エラー: 状態の取得に失敗しました - {str(e)}")
    
    def do_nt(self, arg):
        """現在の神経伝達物質レベルを表示します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            state = self.system.explain_current_state()
            print("\n===== 神経伝達物質レベル =====")
            for nt, level in state['neurotransmitter_levels'].items():
                print(f"{nt}: {level:.2f}")
        except Exception as e:
            self.logger.error(f"神経伝達物質レベル取得エラー: {str(e)}")
            print(f"エラー: 神経伝達物質レベルの取得に失敗しました - {str(e)}")
    
    def do_set_nt(self, arg):
        """神経伝達物質のレベルを設定します: set_nt <type> <level>
        例: set_nt dopamine 0.8"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        args = arg.strip().split()
        if len(args) != 2:
            print("エラー: 正しいフォーマットは 'set_nt <type> <level>' です")
            return
        
        nt_type, level_str = args
        try:
            level = float(level_str)
            if level < 0.0 or level > 1.0:
                print("エラー: レベルは 0.0 から 1.0 の間である必要があります")
                return
            
            self.system.adjust_neurotransmitter(nt_type, level)
            self.logger.info(f"神経伝達物質 {nt_type} のレベルを {level:.2f} に設定しました")
            print(f"神経伝達物質 {nt_type} のレベルを {level:.2f} に設定しました")
        except ValueError:
            print(f"エラー: レベルは数値である必要があります: {level_str}")
        except Exception as e:
            self.logger.error(f"神経伝達物質設定エラー: {str(e)}")
            print(f"エラー: 神経伝達物質の設定に失敗しました - {str(e)}")
    
    def do_drug(self, arg):
        """薬物を適用します: drug <name> [dose]
        例: drug methylphenidate 0.5"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        args = arg.strip().split()
        if not args:
            print("エラー: 薬物名を指定してください")
            return
        
        drug_name = args[0]
        dose = float(args[1]) if len(args) > 1 else 1.0
        
        try:
            if dose < 0.0 or dose > 1.0:
                print("エラー: 用量は 0.0 から 1.0 の間である必要があります")
                return
            
            effects = self.system.apply_drug(drug_name, dose)
            self.logger.info(f"薬物 {drug_name} を用量 {dose:.2f} で適用しました")
            print(f"\n薬物 {drug_name} を用量 {dose:.2f} で適用しました")
            
            if effects:
                print("\n=== 効果 ===")
                for effect, details in effects.items():
                    if effect == 'neurotransmitters':
                        print("\n神経伝達物質への影響:")
                        for nt, change in details.items():
                            direction = "増加" if change > 0 else "減少"
                            print(f"  {nt}: {abs(change):.2f} {direction}")
                    elif effect == 'description':
                        print(f"\n説明: {details}")
        except Exception as e:
            self.logger.error(f"薬物適用エラー: {str(e)}")
            print(f"エラー: 薬物の適用に失敗しました - {str(e)}")
    
    def do_explain(self, arg):
        """現在の行動の説明を生成します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            explanation = self.system.generate_behavior_explanation()
            print("\n===== 行動の説明 =====")
            print(f"要約: {explanation['summary']}")
            print("\n詳細説明:")
            print(explanation['detailed'])
            
            if 'cortical_layers' in explanation:
                print("\n皮質層の活動状態:")
                for layer, activity in explanation['cortical_layers'].items():
                    print(f"  {layer}: {activity}")
        except Exception as e:
            self.logger.error(f"説明生成エラー: {str(e)}")
            print(f"エラー: 説明の生成に失敗しました - {str(e)}")
    
    def do_ask(self, arg):
        """システムに質問します: ask <question>
        例: ask ドーパミンの役割について教えてください"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        if not arg.strip():
            print("エラー: 質問を入力してください")
            return
        
        try:
            response = self.system.get_natural_language_explanation(arg)
            print(f"\n===== 回答 =====\n{response}")
        except Exception as e:
            self.logger.error(f"質問応答エラー: {str(e)}")
            print(f"エラー: 質問への応答に失敗しました - {str(e)}")
    
    def do_gesture(self, arg):
        """指定されたジェスチャーを実行します: gesture <name>
        例: gesture wave"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        if not arg.strip():
            print("エラー: ジェスチャー名を指定してください")
            return
        
        try:
            result = self.system.execute_gesture(arg.strip())
            self.logger.info(f"ジェスチャー '{arg.strip()}' を実行しました")
            print(f"ジェスチャー '{arg.strip()}' を実行しました: {result}")
        except Exception as e:
            self.logger.error(f"ジェスチャー実行エラー: {str(e)}")
            print(f"エラー: ジェスチャーの実行に失敗しました - {str(e)}")
    
    def do_graph(self, arg):
        """神経伝達物質レベルをグラフとして表示します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            state = self.system.explain_current_state()
            nt_levels = state['neurotransmitter_levels']
            
            # グラフの作成
            neurotransmitters = list(nt_levels.keys())
            levels = [nt_levels[nt] for nt in neurotransmitters]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(neurotransmitters, levels, color='skyblue')
            
            # バーの上に値を表示
            for bar, level in zip(bars, levels):
                plt.text(bar.get_x() + bar.get_width()/2, level + 0.02,
                        f'{level:.2f}', ha='center', fontsize=10)
            
            plt.title('現在の神経伝達物質レベル')
            plt.ylabel('レベル (0.0-1.0)')
            plt.ylim(0, 1.1)  # 値の表示のために余白を追加
            plt.tight_layout()
            
            # 出力ディレクトリがなければ作成
            output_dir = 'outputs'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # タイムスタンプ付きでファイル保存
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{output_dir}/neurotransmitters_{timestamp}.png"
            plt.savefig(filename)
            plt.close()
            
            print(f"グラフを保存しました: {filename}")
        except Exception as e:
            self.logger.error(f"グラフ生成エラー: {str(e)}")
            print(f"エラー: グラフの生成に失敗しました - {str(e)}")
    
    def do_config(self, arg):
        """現在の設定を表示します"""
        if not self.system:
            print("エラー: システムが開始されていません")
            return
        
        try:
            config = self.system.get_config()
            print("\n===== 現在の設定 =====")
            
            # システム設定
            if 'system' in config:
                print("\n--- システム設定 ---")
                for key, value in config['system'].items():
                    print(f"{key}: {value}")
            
            # 神経伝達物質設定
            if 'neurotransmitters' in config and 'default_levels' in config['neurotransmitters']:
                print("\n--- デフォルト神経伝達物質レベル ---")
                for nt, level in config['neurotransmitters']['default_levels'].items():
                    print(f"{nt}: {level}")
            
            # ロボットインターフェース設定
            if 'robot_interface' in config:
                print("\n--- ロボットインターフェース設定 ---")
                print(f"シミュレーションモード: {config['robot_interface'].get('simulation_mode', 'N/A')}")
                
                conn = config['robot_interface'].get('connection', {})
                print(f"接続タイプ: {conn.get('type', 'N/A')}")
                print(f"ポート: {conn.get('port', 'N/A')}")
                
                sensors = config['robot_interface'].get('sensors', {})
                enabled_sensors = [s for s, enabled in sensors.items() if enabled]
                print(f"有効なセンサー: {', '.join(enabled_sensors) if enabled_sensors else 'なし'}")
            
            # LLM設定
            if 'llm' in config:
                print("\n--- 言語モデル設定 ---")
                print(f"デフォルトプロバイダー: {config['llm'].get('default_provider', 'N/A')}")
                
                for provider in ['openai', 'anthropic', 'local']:
                    if provider in config['llm']:
                        print(f"\n{provider.capitalize()}設定:")
                        provider_config = config['llm'][provider]
                        print(f"  モデル: {provider_config.get('model', 'N/A')}")
                        print(f"  温度: {provider_config.get('temperature', 'N/A')}")
        except Exception as e:
            self.logger.error(f"設定表示エラー: {str(e)}")
            print(f"エラー: 設定の表示に失敗しました - {str(e)}")
    
    def do_exit(self, arg):
        """プログラムを終了します"""
        print("ExplainableRoboticsシェルを終了します...")
        self.logger.info("シェルを終了します")
        return True
    
    def do_quit(self, arg):
        """プログラムを終了します"""
        return self.do_exit(arg)
    
    def default(self, line):
        print(f"エラー: 不明なコマンド '{line}'")
        print("使用可能なコマンドを表示するには 'help' または '?' を入力してください")

def create_default_config(output_path):
    """デフォルト設定ファイルを作成"""
    config = {
        "system": {
            "data_dir": "./data",
            "knowledge_dir": "./data/knowledge",
            "log_dir": "./logs",
            "explanation_dir": "./data/explanations"
        },
        "neurotransmitters": {
            "default_levels": {
                "acetylcholine": 0.5,
                "dopamine": 0.5,
                "serotonin": 0.5,
                "noradrenaline": 0.5,
                "glutamate": 0.5,
                "gaba": 0.5
            },
            "receptor_sensitivities": {
                "glutamate": {
                    "nmda": 1.0,
                    "ampa": 1.0,
                    "kainate": 1.0,
                    "mglur": 1.0
                },
                "gaba": {
                    "gaba_a": 1.0,
                    "gaba_b": 1.0
                }
            }
        },
        "cortical_model": {
            "layers": ["layer1", "layer2_3", "layer4", "layer5", "layer6"],
            "input_dim": 100,
            "output_dim": 22,
            "activation_functions": {
                "layer1": "relu",
                "layer2_3": "tanh",
                "layer4": "relu",
                "layer5": "tanh",
                "layer6": "tanh"
            },
            "layer_sizes": {
                "layer1": 120,
                "layer2_3": 200,
                "layer4": 150,
                "layer5": 100,
                "layer6": 80
            }
        },
        "robot_interface": {
            "simulation_mode": True,
            "connection": {
                "type": "usb",
                "port": "COM3",
                "baudrate": 115200,
                "timeout": 1.0
            },
            "sensors": {
                "camera": True,
                "imu": True,
                "joint_sensors": True,
                "force_sensors": True
            },
            "motors": {
                "head": True,
                "arms": True,
                "legs": True
            },
            "safety": {
                "max_joint_velocity": 1.0,
                "max_torque": 0.8,
                "emergency_stop_enabled": True
            }
        },
        "llm": {
            "default_provider": "openai",
            "openai": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 1024,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0
            },
            "anthropic": {
                "model": "claude-3-opus-20240229",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "knowledge_base": {
                "enabled": True,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "similarity_top_k": 3
            }
        },
        "drugs": {
            "metabolism_rate": 0.01,
            "max_active_duration": 300,
            "methylphenidate": {
                "targets": {
                    "dopamine": 0.3,
                    "noradrenaline": 0.2
                },
                "strength": 0.7,
                "duration": 240,
                "side_effects": {
                    "acetylcholine": 0.1
                }
            },
            "diazepam": {
                "targets": {
                    "gaba": 0.4
                },
                "strength": 0.6,
                "duration": 180,
                "side_effects": {
                    "acetylcholine": -0.1
                }
            },
            "donepezil": {
                "targets": {
                    "acetylcholine": 0.5
                },
                "strength": 0.8,
                "duration": 300,
                "side_effects": {
                    "dopamine": 0.1
                }
            }
        }
    }
    
    # 必要なディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 設定を保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def main():
    """メイン関数"""
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='ExplainableRobotics コマンドラインインターフェース')
    parser.add_argument('--config', '-c', type=str, help='設定ファイルへのパス')
    parser.add_argument('--create-config', action='store_true', help='デフォルト設定ファイルを作成')
    parser.add_argument('--config-path', type=str, default='config/default_config.json',
                        help='作成する設定ファイルのパス (--create-configと共に使用)')
    parser.add_argument('--debug', action='store_true', help='デバッグモードを有効化')
    
    args = parser.parse_args()
    
    # ロギングの設定
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('explainable_robotics.log'),
            logging.StreamHandler() if args.debug else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger("explainable_robotics")
    
    # デフォルト設定ファイルの作成
    if args.create_config:
        config_path = args.config_path
        try:
            create_default_config(config_path)
            logger.info(f"デフォルト設定ファイルを作成しました: {config_path}")
            print(f"デフォルト設定ファイルを作成しました: {config_path}")
            if not args.config:
                return
        except Exception as e:
            logger.error(f"設定ファイル作成エラー: {str(e)}")
            print(f"エラー: 設定ファイルの作成に失敗しました - {str(e)}")
            return
    
    # シェルの開始
    shell = ExplainableRoboticsShell(args.config)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExplainableRoboticsシェルを終了します...")
        logger.info("キーボード割り込みによりシェルを終了します")
    except Exception as e:
        logger.error(f"シェル実行エラー: {str(e)}")
        print(f"エラー: シェルの実行中にエラーが発生しました - {str(e)}")

if __name__ == "__main__":
    main() 