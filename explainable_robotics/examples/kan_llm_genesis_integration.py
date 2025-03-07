"""
BioKAN-LLM-Genesis 統合サンプル

このスクリプトは、BioKAN大脳皮質モデル、LLMエージェント、Genesis物理エンジンを
統合して説明可能なヒューマノイドロボット制御を実現する例を示します。
"""

import os
import sys
import time
import argparse
import json
import logging
from typing import Dict, Any, Optional
import threading

# 親ディレクトリをパスに追加（パッケージ外から実行する場合）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from explainable_robotics import IntegratedSystem, create_integrated_system
from explainable_robotics.utils.logging import setup_logger

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="BioKAN-LLM-Genesis 統合サンプル")
    
    parser.add_argument("--config", "-c", type=str, default=None,
                       help="設定ファイルのパス")
    parser.add_argument("--log-level", "-l", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="ログレベル")
    parser.add_argument("--goal", "-g", type=str, default="前方の障害物を避けながら移動する",
                       help="ロボットの目標")
    parser.add_argument("--duration", "-d", type=int, default=60,
                       help="実行時間（秒）")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="インタラクティブモード")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="実行結果の出力ファイル")
    
    return parser.parse_args()

def create_default_config() -> Dict[str, Any]:
    """デフォルト設定を作成"""
    return {
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
            }
        },
        "cortical_model": {
            "input_dim": 100,
            "hidden_dim": 256,
            "output_dim": 50,
            "num_layers": 6,
            "learning_rate": 0.001
        },
        "llm": {
            "default_provider": "openai",
            "alternative_providers": ["google", "anthropic"],
            "fallback_enabled": True,
            "temperature": 0.7,
            "max_tokens": 1024
        },
        "robot_interface": {
            "simulation_mode": True
        }
    }

def save_config(config: Dict[str, Any], path: str) -> None:
    """設定をファイルに保存"""
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"設定を保存しました: {path}")
    except Exception as e:
        print(f"設定の保存中にエラーが発生しました: {str(e)}")

def load_config(path: str) -> Dict[str, Any]:
    """設定ファイルを読み込み"""
    if not os.path.exists(path):
        print(f"設定ファイルが見つかりません: {path}")
        return create_default_config()
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"設定を読み込みました: {path}")
        return config
    except Exception as e:
        print(f"設定の読み込み中にエラーが発生しました: {str(e)}")
        return create_default_config()

def interactive_mode(system: IntegratedSystem) -> None:
    """インタラクティブモード"""
    print("\n=== インタラクティブモード ===")
    print("コマンドを入力してください（helpでコマンド一覧、exitで終了）")
    
    while True:
        try:
            command = input("\nコマンド> ").strip()
            
            if command.lower() in ["exit", "quit", "q"]:
                break
                
            elif command.lower() in ["help", "h", "?"]:
                print("コマンド一覧:")
                print("  start - システムを開始")
                print("  stop - システムを停止")
                print("  status - 現在の状態を表示")
                print("  explain [質問] - 説明を取得")
                print("  nt [名前] [値] - 神経伝達物質レベルを設定")
                print("  exit - 終了")
                
            elif command.lower() == "start":
                print("システムを開始します...")
                system.start()
                
            elif command.lower() == "stop":
                print("システムを停止します...")
                system.stop()
                
            elif command.lower() == "status":
                nt_levels = system.nt_system.get_all_levels()
                print("\n=== システムステータス ===")
                print(f"実行状態: {'実行中' if system.running else '停止中'}")
                print("\n神経伝達物質レベル:")
                for nt, level in nt_levels.items():
                    print(f"  {nt}: {level:.2f}")
                    
                if hasattr(system, "last_action") and system.last_action:
                    print("\n最後の行動:")
                    print(f"  種類: {system.last_action.get('type', 'unknown')}")
                    print(f"  パラメータ: {system.last_action.get('parameters', {})}")
                    print(f"  確信度: {system.last_action.get('confidence', 0.0):.2f}")
                    
            elif command.lower().startswith("explain"):
                query = command[7:].strip() if len(command) > 7 else "現在の状態を説明してください"
                print(f"\n質問: {query}")
                
                explanation = system.get_natural_language_explanation(query)
                print(f"\n説明:\n{explanation}")
                
            elif command.lower().startswith("nt "):
                parts = command.split()
                if len(parts) >= 3:
                    nt_name = parts[1]
                    try:
                        nt_value = float(parts[2])
                        if 0.0 <= nt_value <= 1.0:
                            system.nt_system.set_level(nt_name, nt_value)
                            print(f"{nt_name}のレベルを{nt_value:.2f}に設定しました")
                        else:
                            print("神経伝達物質のレベルは0.0～1.0の範囲で指定してください")
                    except ValueError:
                        print("無効な値です。数値を指定してください")
                else:
                    print("使用法: nt <神経伝達物質名> <値>")
                    
            else:
                print(f"不明なコマンド: {command}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            
    print("インタラクティブモードを終了します")

def main():
    """メイン処理"""
    # 引数の解析
    args = parse_args()
    
    # ログの設定
    log_level = getattr(logging, args.log_level)
    setup_logger(level=log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("BioKAN-LLM-Genesis 統合サンプルを開始します")
    
    try:
        # 設定の読み込み
        config_path = args.config
        if config_path:
            config = load_config(config_path)
        else:
            config = create_default_config()
            config_path = "config.json"
            save_config(config, config_path)
            
        # 統合システムの作成
        logger.info("統合システムを作成中...")
        system = create_integrated_system(config_path)
        
        # ロボットの目標を設定
        logger.info(f"ロボットの目標: {args.goal}")
        goal = args.goal
        
        # システムを開始
        system.start()
        logger.info("システムを開始しました")
        
        try:
            if args.interactive:
                # インタラクティブモード
                interactive_mode(system)
            else:
                # 指定された時間だけ実行
                logger.info(f"{args.duration}秒間実行します...")
                time.sleep(args.duration)
                
                # 状態の説明を取得
                explanation = system.get_natural_language_explanation(
                    f"目標「{goal}」に対する行動と結果を説明してください"
                )
                logger.info(f"\n説明:\n{explanation}")
                
                # 結果を保存
                if args.output:
                    try:
                        result = {
                            "goal": goal,
                            "duration": args.duration,
                            "explanation": explanation,
                            "timestamp": time.time()
                        }
                        
                        directory = os.path.dirname(args.output)
                        if directory and not os.path.exists(directory):
                            os.makedirs(directory)
                            
                        with open(args.output, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, ensure_ascii=False)
                            
                        logger.info(f"結果を保存しました: {args.output}")
                    except Exception as e:
                        logger.error(f"結果の保存中にエラーが発生しました: {str(e)}")
                
        finally:
            # システムを停止
            system.stop()
            logger.info("システムを停止しました")
            
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        return 1
        
    logger.info("BioKAN-LLM-Genesis 統合サンプルを終了します")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 