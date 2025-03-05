#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コマンドラインインターフェース

explainable_roboticsのコマンドラインツールを提供します。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .utils.logging import get_logger, setup_global_logging
from .controller.robot_controller import RobotController
from .demos.humanoid_demo import main as run_humanoid_demo

# ロガーの設定
logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析します。"""
    parser = argparse.ArgumentParser(
        description="Explainable Robotics - 神経科学的に妥当なヒューマノイドロボット制御フレームワーク",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.json",
        help="設定ファイルのパス"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="対話モードで実行"
    )
    
    parser.add_argument(
        "--demo", 
        type=str, 
        choices=["humanoid", "cortical", "visualizer"], 
        help="実行するデモの種類"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベル"
    )
    
    parser.add_argument(
        "--log-file", 
        type=str, 
        help="ログファイルのパス（指定しない場合は標準出力のみ）"
    )
    
    parser.add_argument(
        "--no-color", 
        action="store_true",
        help="ログ出力のカラー表示を無効化"
    )
    
    parser.add_argument(
        "--version", 
        action="store_true",
        help="バージョン情報を表示して終了"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """設定ファイルを読み込みます。"""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"設定ファイル {config_path} が見つかりません。デフォルト設定を使用します。")
            return {}
            
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        return {}

def setup_environment(config: Dict[str, Any]) -> None:
    """環境変数を設定します。"""
    if "environment" in config:
        for key, value in config["environment"].items():
            os.environ[key] = str(value)
            logger.debug(f"環境変数を設定: {key}={value}")

def show_version() -> None:
    """バージョン情報を表示します。"""
    from . import __version__, __author__, __license__
    print(f"Explainable Robotics v{__version__}")
    print(f"作者: {__author__}")
    print(f"ライセンス: {__license__}")
    sys.exit(0)

def interactive_mode(config: Dict[str, Any]) -> None:
    """対話モードを実行します。"""
    print("=" * 80)
    print("Explainable Robotics 対話モード")
    print("=" * 80)
    print("使用可能なコマンド:")
    print("  help       - このヘルプメッセージを表示")
    print("  start      - ロボットコントローラーを起動")
    print("  stop       - ロボットコントローラーを停止")
    print("  status     - 現在のシステム状態を表示")
    print("  neurotx    - 神経伝達物質レベルを設定")
    print("  goal       - ロボットの目標を設定")
    print("  demo       - デモを実行")
    print("  exit       - 対話モードを終了")
    print("=" * 80)
    
    controller: Optional[RobotController] = None
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == "exit":
                if controller and controller.is_running:
                    controller.stop()
                print("システムを終了します...")
                break
                
            elif command == "help":
                print("使用可能なコマンド:")
                print("  help       - このヘルプメッセージを表示")
                print("  start      - ロボットコントローラーを起動")
                print("  stop       - ロボットコントローラーを停止")
                print("  status     - 現在のシステム状態を表示")
                print("  neurotx    - 神経伝達物質レベルを設定")
                print("  goal       - ロボットの目標を設定")
                print("  demo       - デモを実行")
                print("  exit       - 対話モードを終了")
                
            elif command == "start":
                if controller and controller.is_running:
                    print("コントローラーは既に実行中です。")
                else:
                    robot_config = config.get("robot", {})
                    controller = RobotController(
                        robot_name=robot_config.get("name", "humanoid"),
                        model_path=robot_config.get("model_path", "models/humanoid.urdf"),
                        biokan_config=config.get("biokan", {}),
                        gemini_config=config.get("gemini", {}),
                        visualizer_config=config.get("visualizer", {}),
                    )
                    controller.start()
                    print("ロボットコントローラーを起動しました。")
                    
            elif command == "stop":
                if controller and controller.is_running:
                    controller.stop()
                    print("ロボットコントローラーを停止しました。")
                else:
                    print("コントローラーは実行されていません。")
                    
            elif command == "status":
                if controller:
                    print(f"ロボット名: {controller.robot_name}")
                    print(f"実行中: {controller.is_running}")
                    if controller.state:
                        print(f"現在の状態: {controller.state.get('status', '不明')}")
                        print(f"目標: {controller.state.get('goals', '設定なし')}")
                        print("神経伝達物質レベル:")
                        neurotx = controller.biokan.neurotransmitters if controller.biokan else {}
                        for name, level in neurotx.items():
                            print(f"  {name}: {level:.2f}")
                else:
                    print("コントローラーが初期化されていません。")
                    
            elif command == "neurotx":
                if not controller or not controller.biokan:
                    print("BioKANモデルが初期化されていません。まず'start'コマンドを実行してください。")
                    continue
                    
                print("設定する神経伝達物質を選択:")
                print("1: dopamine (意欲・報酬)")
                print("2: serotonin (情緒安定)")
                print("3: acetylcholine (集中・記憶)")
                print("4: norepinephrine (覚醒・興奮)")
                print("5: gaba (抑制・リラックス)")
                print("6: glutamate (興奮・学習)")
                
                try:
                    neurotx_choice = int(input("選択 (1-6): ").strip())
                    if neurotx_choice < 1 or neurotx_choice > 6:
                        print("無効な選択です。")
                        continue
                        
                    neurotx_names = ["dopamine", "serotonin", "acetylcholine", 
                                    "norepinephrine", "gaba", "glutamate"]
                    neurotx_name = neurotx_names[neurotx_choice - 1]
                    
                    level = float(input(f"{neurotx_name}のレベル (0.0-1.0): ").strip())
                    if level < 0.0 or level > 1.0:
                        print("レベルは0.0から1.0の範囲で指定してください。")
                        continue
                        
                    controller.biokan.set_neurotransmitter(neurotx_name, level)
                    print(f"{neurotx_name}のレベルを{level:.2f}に設定しました。")
                    
                except ValueError:
                    print("数値を入力してください。")
                    
            elif command == "goal":
                if not controller:
                    print("コントローラーが初期化されていません。まず'start'コマンドを実行してください。")
                    continue
                    
                goal = input("新しい目標を入力: ").strip()
                if goal:
                    controller.set_goals(goal)
                    print(f"目標を設定: {goal}")
                else:
                    print("目標は空にできません。")
                    
            elif command == "demo":
                print("実行するデモを選択:")
                print("1: humanoid - ヒューマノイドロボットデモ")
                print("2: cortical - 大脳皮質モデルデモ")
                print("3: visualizer - 可視化デモ")
                
                try:
                    demo_choice = int(input("選択 (1-3): ").strip())
                    if demo_choice == 1:
                        print("ヒューマノイドロボットデモを実行します...")
                        run_humanoid_demo()
                    elif demo_choice == 2:
                        print("大脳皮質モデルデモを実行します...")
                        # TODO: 大脳皮質モデルデモの実装
                        print("未実装です。")
                    elif demo_choice == 3:
                        print("可視化デモを実行します...")
                        # TODO: 可視化デモの実装
                        print("未実装です。")
                    else:
                        print("無効な選択です。")
                except ValueError:
                    print("数値を入力してください。")
                    
            else:
                print(f"未知のコマンド: {command}")
                print("'help'と入力するとコマンド一覧が表示されます。")
                
        except KeyboardInterrupt:
            print("\nCtrl+Cが押されました。終了します...")
            if controller and controller.is_running:
                controller.stop()
            break
            
        except Exception as e:
            logger.error(f"エラーが発生しました: {e}")

def main() -> None:
    """メイン関数"""
    args = parse_args()
    
    # バージョン表示
    if args.version:
        show_version()
        
    # ログの設定
    setup_global_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        use_color=not args.no_color
    )
    
    # 設定の読み込み
    config = load_config(args.config)
    
    # 環境の設定
    setup_environment(config)
    
    logger.info("Explainable Robotics を起動しています...")
    
    try:
        # 実行モードの決定
        if args.demo:
            if args.demo == "humanoid":
                logger.info("ヒューマノイドロボットデモを実行します...")
                run_humanoid_demo()
            elif args.demo == "cortical":
                logger.info("大脳皮質モデルデモを実行します...")
                # TODO: 大脳皮質モデルデモの実装
                logger.warning("大脳皮質モデルデモは未実装です。")
            elif args.demo == "visualizer":
                logger.info("可視化デモを実行します...")
                # TODO: 可視化デモの実装
                logger.warning("可視化デモは未実装です。")
        elif args.interactive:
            interactive_mode(config)
        else:
            # デフォルト動作
            logger.info("対話モードを開始します...")
            interactive_mode(config)
            
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断を検出しました。終了します...")
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}", exc_info=True)
    finally:
        logger.info("Explainable Robotics を終了します。")
        
if __name__ == "__main__":
    main() 