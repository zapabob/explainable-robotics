"""
統合デモスクリプト

BioKAN、Gemini Pro、Genesisを統合して、ヒューマノイドロボットのデモを実行します。
このデモでは、以下の機能を実装します：

1. 統合脳システムの初期化と実行
2. Genesisを使用したヒューマノイドロボットの3D可視化
3. 神経伝達物質レベルと感情状態の可視化
4. 行動の可視化と説明
5. 薬物効果のシミュレーション
"""

import os
import time
import json
import numpy as np
from typing import Dict, Any, Optional
import threading
import queue
import argparse

from ..core.integrated_brain import IntegratedBrain
from ..visualization.genesis_visualizer import GenesisVisualizer
from ..utils.logging import get_logger

logger = get_logger(__name__)

class IntegratedDemo:
    """
    統合デモクラス
    
    BioKAN、Gemini Pro、Genesisを統合して、ヒューマノイドロボットのデモを実行します。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config_path: Optional[str] = None,
        window_title: str = "ExplainableRobotics - 統合デモ",
        width: int = 1280,
        height: int = 720,
        show_ui: bool = True
    ):
        """
        初期化
        
        Args:
            api_key: Google AI Studio APIキー
            config_path: 設定ファイルのパス
            window_title: ウィンドウのタイトル
            width: ウィンドウの幅
            height: ウィンドウの高さ
            show_ui: UIを表示するかどうか
        """
        logger.info("統合デモを初期化しています...")
        
        # 設定の読み込み
        self.config = self._load_config(config_path)
        
        # APIキーの設定
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or self.config.get("api_key")
        if not self.api_key:
            error_msg = "Google APIキーが必要です"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 統合脳システムの初期化
        logger.info("統合脳システムを初期化しています...")
        self.brain = IntegratedBrain(
            api_key=self.api_key,
            cortical_config=self.config.get("cortical"),
            gemini_config=self.config.get("gemini")
        )
        
        # 可視化システムの初期化
        logger.info("可視化システムを初期化しています...")
        self.visualizer = GenesisVisualizer(
            window_title=window_title,
            width=width,
            height=height,
            show_ui=show_ui
        )
        
        # 更新キュー
        self.update_queue = queue.Queue()
        
        # 状態
        self.is_running = False
        self.thread = None
        
        # デモパラメータ
        self.demo_params = {
            "update_interval": 0.1,  # 更新間隔（秒）
            "exploration_rate": 0.2,  # 探索率
            "max_speed": 0.5,  # 最大移動速度
            "reward_scale": 1.0,  # 報酬のスケール
            "drug_interval": 30.0,  # 薬物投与間隔（秒）
            "last_drug_time": 0.0  # 最後の薬物投与時刻
        }
        
        # 薬物リスト
        self.drugs = [
            {"name": "dopamine_agonist", "dose": 0.8},
            {"name": "serotonin_agonist", "dose": 0.7},
            {"name": "acetylcholine_agonist", "dose": 0.6},
            {"name": "noradrenaline_agonist", "dose": 0.5},
            {"name": "gaba_agonist", "dose": 0.4},
            {"name": "glutamate_agonist", "dose": 0.3}
        ]
        
        logger.info("統合デモの初期化が完了しました")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        設定ファイルの読み込み
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定辞書
        """
        config = {}
        
        # デフォルト設定
        default_config = {
            "cortical": {
                "input_dim": 100,
                "hidden_dim": 256,
                "output_dim": 50,
                "num_layers": 6,
                "learning_rate": 0.001
            },
            "gemini": {
                "model_name": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "visualization": {
                "window_title": "ExplainableRobotics - 統合デモ",
                "width": 1280,
                "height": 720,
                "show_ui": True
            }
        }
        
        # 設定ファイルの読み込み
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    config.update(file_config)
                logger.info(f"設定ファイルを読み込みました: {config_path}")
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
        
        # デフォルト設定で補完
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config[key]}
        
        return config
    
    def start(self):
        """デモの開始"""
        if self.is_running:
            logger.warning("デモは既に実行中です")
            return
        
        self.is_running = True
        
        # 更新スレッドの開始
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()
        
        # 可視化の開始
        self.visualizer.start()
        
        logger.info("デモを開始しました")
    
    def stop(self):
        """デモの停止"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        self.visualizer.stop()
        
        logger.info("デモを停止しました")
    
    def _update_loop(self):
        """更新ループ"""
        last_update = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                dt = current_time - last_update
                
                if dt >= self.demo_params["update_interval"]:
                    # 感覚入力の生成
                    sensory_input = self._generate_sensory_input()
                    
                    # 報酬の計算
                    reward = self._compute_reward(sensory_input)
                    
                    # 脳の更新
                    result = self.brain.process_input(sensory_input, reward)
                    
                    # 行動の実行
                    self._execute_action(result["action"])
                    
                    # 可視化の更新
                    self._update_visualization(result)
                    
                    # 薬物効果のシミュレーション
                    self._simulate_drug_effects(current_time)
                    
                    last_update = current_time
                
                time.sleep(0.016)  # 約60FPS
                
            except Exception as e:
                logger.error(f"更新ループでエラーが発生しました: {e}")
                time.sleep(1.0)
    
    def _generate_sensory_input(self) -> Dict[str, Any]:
        """
        感覚入力の生成
        
        Returns:
            感覚入力データ
        """
        # 現在の位置を取得
        current_pos = self.visualizer.human.position if self.visualizer.human else (0, 0, 0)
        
        # 関節角度を取得（実際のロボットでは、センサーから取得）
        joint_angles = np.random.randn(20) * 0.1  # ノイズを含む関節角度
        
        # 感覚入力の作成
        sensory_input = {
            "position": current_pos,
            "joint_angles": joint_angles.tolist(),
            "time": time.time()
        }
        
        return sensory_input
    
    def _compute_reward(self, sensory_input: Dict[str, Any]) -> float:
        """
        報酬の計算
        
        Args:
            sensory_input: 感覚入力
            
        Returns:
            報酬値
        """
        # 簡単な報酬関数の例：
        # - 原点からの距離に基づく報酬（近いほど良い）
        # - 安定した姿勢に対する報酬
        
        position = np.array(sensory_input["position"])
        distance = np.linalg.norm(position)
        
        # 距離に基づく報酬（-1〜1の範囲）
        distance_reward = np.exp(-distance) * 2.0 - 1.0
        
        # 姿勢に基づく報酬（関節角度の標準偏差が小さいほど良い）
        joint_angles = np.array(sensory_input["joint_angles"])
        stability_reward = np.exp(-np.std(joint_angles)) * 2.0 - 1.0
        
        # 総合報酬
        reward = (distance_reward + stability_reward) / 2.0
        reward *= self.demo_params["reward_scale"]
        
        return reward
    
    def _execute_action(self, action: Dict[str, Any]):
        """
        行動の実行
        
        Args:
            action: 行動データ
        """
        # 行動の正規化（最大速度で制限）
        max_speed = self.demo_params["max_speed"]
        normalized_action = {
            "x": np.clip(action["x"], -max_speed, max_speed),
            "y": np.clip(action["y"], -max_speed, max_speed),
            "z": np.clip(action["z"], -max_speed, max_speed),
            "reason": action["reason"]
        }
        
        # 可視化の更新
        self.visualizer.update_action(normalized_action)
    
    def _update_visualization(self, result: Dict[str, Any]):
        """
        可視化の更新
        
        Args:
            result: 脳の処理結果
        """
        # 脳の状態を更新
        brain_state = {
            "neurotransmitter_levels": result["internal_state"]["neurotransmitter_levels"],
            "emotional_state": result["internal_state"]["emotional_state"],
            "cortical_activity": result["cortical_output"]["activity"]
        }
        self.visualizer.update_brain_state(brain_state)
    
    def _simulate_drug_effects(self, current_time: float):
        """
        薬物効果のシミュレーション
        
        Args:
            current_time: 現在時刻
        """
        # 一定間隔で薬物を投与
        if current_time - self.demo_params["last_drug_time"] >= self.demo_params["drug_interval"]:
            # ランダムに薬物を選択
            drug = np.random.choice(self.drugs)
            
            # 薬物効果のシミュレーション
            new_levels = self.brain.simulate_drug_effect(drug["name"], drug["dose"])
            
            # 結果をログに記録
            logger.info(f"薬物を投与しました: {drug['name']} (用量: {drug['dose']:.2f})")
            logger.info("神経伝達物質レベル:")
            for nt, level in new_levels.items():
                logger.info(f"  {nt}: {level:.2f}")
            
            self.demo_params["last_drug_time"] = current_time


def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="ExplainableRobotics 統合デモ")
    parser.add_argument("--api-key", help="Google AI Studio APIキー")
    parser.add_argument("--config", help="設定ファイルのパス")
    parser.add_argument("--width", type=int, default=1280, help="ウィンドウの幅")
    parser.add_argument("--height", type=int, default=720, help="ウィンドウの高さ")
    parser.add_argument("--no-ui", action="store_true", help="UIを非表示にする")
    args = parser.parse_args()
    
    try:
        # デモの作成と実行
        demo = IntegratedDemo(
            api_key=args.api_key,
            config_path=args.config,
            width=args.width,
            height=args.height,
            show_ui=not args.no_ui
        )
        
        # デモの開始
        demo.start()
        
    except KeyboardInterrupt:
        logger.info("デモを終了します...")
        demo.stop()
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()