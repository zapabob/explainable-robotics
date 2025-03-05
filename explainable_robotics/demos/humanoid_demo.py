#!/usr/bin/env python3
"""ヒューマノイドロボットデモ

BioKANモデル、マルチLLMエージェント（OpenAI、Claude、Gemini、Google AI Studio、ローカルLLMに対応）、
Genesisビジュアライザーを組み合わせたヒューマノイドロボットのデモスクリプト。
"""

import os
import sys
import time
import argparse
import json
import traceback
from typing import Dict, Any, Optional
import logging

# 親ディレクトリをモジュール検索パスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ロボットコントローラーのインポート
from explainable_robotics.controller.robot_controller import RobotController
from explainable_robotics.utils.logging import setup_global_logging, get_logger

# ロガーの設定
logger = get_logger(__name__)


def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='ヒューマノイドロボットデモ')

    # 基本設定
    parser.add_argument('--model-path', type=str, default=None,
                        help='ロボットモデルのパス (URDF形式)')

    parser.add_argument('--goal', type=str, default="部屋を探索して挨拶する",
                        help='ロボットの目標（自然言語）')

    parser.add_argument('--duration', type=int, default=60,
                        help='デモの実行時間（秒）')

    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='ログレベル')

    parser.add_argument('--no-visualization', action='store_true',
                        help='可視化なしで実行')

    parser.add_argument('--save-state', type=str, default=None,
                        help='ロボットの状態を保存するファイルパス')

    parser.add_argument('--load-state', type=str, default=None,
                        help='ロボットの状態を読み込むファイルパス')

    # LLM設定グループ
    llm_group = parser.add_argument_group('LLM設定')
    llm_group.add_argument('--llm-provider', type=str, default='gemini',
                           choices=['openai', 'claude', 'gemini', 'vertex', 'local'],
                           help='使用するLLMプロバイダー')
    llm_group.add_argument('--api-key', type=str, help='LLM APIキー')
    llm_group.add_argument('--model-name', type=str, help='LLMモデル名')
    llm_group.add_argument('--temperature', type=float, default=0.7, 
                           help='生成温度 (0.0-1.0)')
    llm_group.add_argument('--local-model-path', type=str, 
                           help='ローカルLLMモデルパス（ローカルLLM使用時）')
    llm_group.add_argument('--credentials-path', type=str, 
                           help='Google Cloud認証情報ファイルパス（Vertex AI使用時）')
    llm_group.add_argument('--project-id', type=str, 
                           help='Google CloudプロジェクトID（Vertex AI使用時）')

    parser.add_argument('--config', type=str, default=None,
                      help='設定ファイルのパス')

    return parser.parse_args()


def setup_logging(log_level):
    """ロギングの設定"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    # グローバルロギング設定
    try:
        setup_global_logging(level=numeric_level, log_file="logs/humanoid_demo.log")
    except Exception as e:
        print(f"ログ設定エラー: {e}")
        # 標準のロギング設定にフォールバック
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )


def load_config(args):
    """設定の読み込み"""
    config = {
        "robot": {},
        "biokan": {},
        "llm": {},  # gemini_configからllm_configに変更
        "visualizer": {}
    }
    
    # 設定ファイルからの読み込み
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                
            # 設定をマージ
            for section in config.keys():
                if section in file_config:
                    config[section].update(file_config[section])
                    
            logger.info(f"設定ファイル {args.config} を読み込みました")
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
    
    # コマンドライン引数で上書き
    if args.model_path:
        config["robot"]["model_path"] = args.model_path
        
    # LLM設定をコマンドライン引数から更新
    if args.llm_provider:
        config["llm"]["provider"] = args.llm_provider
    if args.api_key:
        config["llm"]["api_key"] = args.api_key
    if args.model_name:
        config["llm"]["model_name"] = args.model_name
    if args.temperature:
        config["llm"]["temperature"] = args.temperature
    if args.local_model_path:
        config["llm"]["local_model_path"] = args.local_model_path
    if args.credentials_path:
        config["llm"]["credentials_path"] = args.credentials_path
    if args.project_id:
        config["llm"]["project_id"] = args.project_id
        
    # 可視化設定
    if args.no_visualization:
        config["visualizer"]["show_ui"] = False
        
    return config


def setup_dummy_sensor_data(controller):
    """ダミーセンサーデータのセットアップ"""
    # カメラデータのシミュレーション
    controller.state["sensor_data"] = {
        "camera": {
            "resolution": [1280, 720],
            "field_of_view": 90,
            "detected_objects": [
                {"type": "person", "distance": 1.5, "position": [0.5, 0, 1.0]},
                {"type": "chair", "distance": 2.0, "position": [-1.0, 0, 0.5]},
                {"type": "table", "distance": 3.0, "position": [2.0, 0, 1.2]}
            ]
        },
        "lidar": {
            "range": 10.0,
            "resolution": 1.0,
            "readings": [2.5, 3.2, 1.8, 4.5, 5.0, 0.8, 1.2]
        },
        "microphone": {
            "detected_speech": "",
            "noise_level": 0.2,
            "direction": None
        },
        "imu": {
            "acceleration": [0.0, 0.0, 9.8],
            "angular_velocity": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0]
        },
        "position": [0.0, 0.0, 1.2],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "joint_angles": {"head": 0.0, "neck": 0.0, "shoulder_right": 0.0},
        "joint_velocities": {"head": 0.0, "neck": 0.0, "shoulder_right": 0.0},
        "battery_level": 0.85
    }


def update_sensor_data(controller):
    """センサーデータの更新"""
    # 実際の実装では、実際のセンサーからデータを取得
    # ここでは、簡単のためにランダムな変動を加える
    import random
    
    # 人の距離を変動させる
    if "camera" in controller.state["sensor_data"]:
        if "detected_objects" in controller.state["sensor_data"]["camera"]:
            for obj in controller.state["sensor_data"]["camera"]["detected_objects"]:
                if obj["type"] == "person":
                    # 人との距離をランダムに変動
                    obj["distance"] += random.uniform(-0.1, 0.1)
                    obj["distance"] = max(0.5, min(5.0, obj["distance"]))
                    
                    # 位置も少し変動
                    obj["position"][0] += random.uniform(-0.05, 0.05)
                    obj["position"][2] += random.uniform(-0.02, 0.02)
    
    # LIDARの読み取り値を更新
    if "lidar" in controller.state["sensor_data"]:
        if "readings" in controller.state["sensor_data"]["lidar"]:
            readings = controller.state["sensor_data"]["lidar"]["readings"]
            for i in range(len(readings)):
                readings[i] += random.uniform(-0.2, 0.2)
                readings[i] = max(0.5, min(10.0, readings[i]))
    
    # バッテリーレベルを少し減らす
    if "battery_level" in controller.state["sensor_data"]:
        controller.state["sensor_data"]["battery_level"] -= random.uniform(0.0, 0.001)
        controller.state["sensor_data"]["battery_level"] = max(0.0, controller.state["sensor_data"]["battery_level"])
        
    # 時々、音声を検出
    if random.random() < 0.05:  # 5%の確率で音声検出
        controller.state["sensor_data"]["microphone"]["detected_speech"] = random.choice([
            "こんにちは、ロボットさん",
            "どちらに行くの？",
            "今何をしているの？",
            ""  # 何も検出しない場合もある
        ])
    else:
        controller.state["sensor_data"]["microphone"]["detected_speech"] = ""


def log_final_results(controller):
    """最終結果をログに記録"""
    logger.info("------ デモ実行結果 ------")
    logger.info(f"ロボット名: {controller.robot_name}")
    logger.info(f"実行時間: {controller.runtime:.2f}秒")
    logger.info(f"目標: {controller.goals}")
    
    # 行動履歴
    if hasattr(controller, "action_history") and controller.action_history:
        logger.info(f"実行した行動数: {len(controller.action_history)}")
        latest_actions = controller.action_history[-3:]  # 最新の3つの行動
        logger.info("最近の行動:")
        for i, action in enumerate(latest_actions):
            logger.info(f"  {i+1}. {action.get('type', 'unknown')}: {action.get('description', '説明なし')}")
    
    # 感情状態
    if hasattr(controller, "biokan") and controller.biokan:
        neurotransmitters = controller.biokan.get_neurotransmitter_levels()
        logger.info("神経伝達物質レベル:")
        for nt_name, level in neurotransmitters.items():
            logger.info(f"  {nt_name}: {level:.2f}")


def main():
    """メイン関数"""
    # 引数の解析
    args = parse_arguments()
    
    # ロギングの設定
    setup_logging(args.log_level)
    
    # 設定の読み込み
    config = load_config(args)
    
    # ロボットコントローラーの初期化
    logger.info("ロボットコントローラを初期化中...")
    controller = RobotController(
        robot_name="humanoid",
        model_path=config["robot"].get("model_path"),
        biokan_config=config["biokan"],
        llm_config=config["llm"],  # gemini_configからllm_configに変更
        visualizer_config=config["visualizer"],
        log_actions=config["robot"].get("log_actions", True),
        action_log_path=config["robot"].get("action_log_path")
    )
    
    # ロボットの起動
    logger.info("ロボットを起動中...")
    if not controller.start():
        logger.error("ロボットの起動に失敗しました")
        return 1
        
    # 状態の読み込み（指定されている場合）
    if args.load_state and os.path.exists(args.load_state):
        logger.info(f"状態を読み込み中: {args.load_state}")
        try:
            controller.load_state(args.load_state)
        except Exception as e:
            logger.error(f"状態の読み込みに失敗しました: {e}")
    
    # ダミーセンサーデータのセットアップ
    setup_dummy_sensor_data(controller)
    
    # 目標の設定
    logger.info(f"目標を設定: {args.goal}")
    controller.set_goals(args.goal)
    
    # 使用しているLLMプロバイダーの表示
    logger.info(f"LLMプロバイダー: {config['llm'].get('provider', 'gemini')}")
    if config['llm'].get('model_name'):
        logger.info(f"LLMモデル: {config['llm']['model_name']}")
    
    # デモのメインループ
    try:
        logger.info(f"デモを開始します（実行時間: {args.duration}秒）")
        start_time = time.time()
        update_interval = 0.1  # 更新間隔（秒）
        
        while time.time() - start_time < args.duration:
            # センサーデータの更新
            update_sensor_data(controller)
            
            # コントローラの更新
            controller.update()
            
            # 少し待機
            time.sleep(update_interval)
            
        elapsed_time = time.time() - start_time
        logger.info(f"デモが完了しました（実際の実行時間: {elapsed_time:.2f}秒）")
        
        # 状態の保存（指定されている場合）
        if args.save_state:
            save_dir = os.path.dirname(args.save_state)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            logger.info(f"状態を保存中: {args.save_state}")
            try:
                controller.save_state(args.save_state)
            except Exception as e:
                logger.error(f"状態の保存に失敗しました: {e}")
                
        # 最終結果のログ記録
        log_final_results(controller)
        
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断を検出しました")
    except Exception as e:
        logger.error(f"実行中にエラーが発生しました: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # ロボットの停止
        logger.info("ロボットを停止中...")
        controller.stop()
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 