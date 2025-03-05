"""
ヒューマノイドロボットのデモスクリプト

Gemini Proを頭脳として使用し、BioKANフレームワークによる強化学習を行います。
Genesisライブラリを使用して3D可視化を行います。
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import torch
import numpy as np
import traceback

from ..llm.gemini_agent import GeminiAgent
from ..cortical.model import CorticalModel, create_cortical_model
from ..control.humanoid import HumanoidController
from ..visualization.genesis_visualizer import GenesisVisualizer
from ..utils.logging import get_logger

logger = get_logger(__name__)

class HumanoidDemo:
    """ヒューマノイドロボットのデモクラス"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cortical_config: Optional[Dict[str, Any]] = None,
        visualize: bool = True
    ):
        """
        初期化
        
        Args:
            api_key: Google AI Studio APIキー
            cortical_config: 大脳皮質モデルの設定
            visualize: 可視化を行うかどうか
        """
        logger.info("ヒューマノイドロボットデモを初期化しています...")
        
        # APIキーの確認
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            error_msg = "GOOGLE_API_KEY環境変数が設定されていません"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Geminiエージェントの初期化
            logger.info("Geminiエージェントを初期化しています...")
            self.agent = GeminiAgent(api_key=self.api_key)
            
            # 大脳皮質モデルの設定
            self.cortical_config = cortical_config or {
                "input_dim": 100,
                "hidden_dim": 256,
                "output_dim": 50,
                "use_glia": True,
                "use_neuromodulation": True,
                "dropout": 0.1
            }
            
            # 大脳皮質モデルの作成
            logger.info("大脳皮質モデルを作成しています...")
            self.cortical_model = create_cortical_model(**self.cortical_config)
            
            # ヒューマノイドコントローラーの初期化
            logger.info("ヒューマノイドコントローラーを初期化しています...")
            self.controller = HumanoidController()
            
            # 学習状態の初期化
            self.state = {
                "position": np.zeros(3),
                "velocity": np.zeros(3),
                "joint_angles": np.zeros(20),
                "reward": 0.0
            }
            
            # 可視化の設定
            self.visualize = visualize
            if self.visualize:
                logger.info("Genesis可視化を初期化しています...")
                self.visualizer = GenesisVisualizer()
            
            logger.info("初期化が完了しました")
            
        except Exception as e:
            logger.error(f"初期化に失敗しました: {e}")
            traceback.print_exc()
            sys.exit(1)
    
    def run_demo(self, num_steps: int = 100):
        """
        デモの実行
        
        Args:
            num_steps: 実行ステップ数
        """
        logger.info(f"デモを開始します（ステップ数: {num_steps}）")
        
        try:
            for step in range(num_steps):
                logger.info(f"ステップ {step+1}/{num_steps} を実行しています...")
                
                # 現在の状態を取得
                current_state = self.controller.get_state()
                
                # 状態をGeminiエージェントに送信
                context = {
                    "step": step,
                    "state": current_state,
                    "reward": self.state["reward"]
                }
                
                # エージェントからの行動指示を取得
                logger.info("Geminiエージェントから行動指示を取得中...")
                action_text = self.agent.process_input(
                    "現在の状態に基づいて、次の行動を決定してください。",
                    context
                )
                
                # 行動を解析して実行
                action = self._parse_action(action_text)
                logger.info(f"行動: {action}")
                self._execute_action(action)
                
                # 報酬の計算と学習
                reward = self._calculate_reward()
                self._update_model(reward)
                
                # 状態の更新
                self.state["reward"] = reward
                
                # 可視化の更新
                if self.visualize:
                    logger.info("可視化を更新しています...")
                    self.visualizer.update_robot_state(self.state)
                    self.visualizer.render()
                
                # 進捗の表示
                if (step + 1) % 10 == 0 or step == 0:
                    logger.info(f"ステップ {step + 1}/{num_steps} 完了")
                    logger.info(f"報酬: {reward:.2f}")
                    
                    # スクリーンショットの保存
                    if self.visualize:
                        screenshot_path = f"screenshot_step_{step+1}.png"
                        self.visualizer.save_screenshot(screenshot_path)
                        logger.info(f"スクリーンショットを保存しました: {screenshot_path}")
                
                # 短い待機時間
                time.sleep(0.5)
        
        except KeyboardInterrupt:
            logger.info("ユーザーによって中断されました")
        except Exception as e:
            logger.error(f"デモの実行中にエラーが発生しました: {e}")
            traceback.print_exc()
        finally:
            # 可視化環境のクリーンアップ
            if self.visualize:
                logger.info("可視化環境をクリーンアップしています...")
                self.visualizer.close()
            
            logger.info("デモが終了しました")
    
    def _parse_action(self, action_text: str) -> Dict[str, Any]:
        """
        エージェントからの行動指示を解析
        
        Args:
            action_text: 行動指示テキスト
            
        Returns:
            解析された行動
        """
        logger.debug(f"行動テキスト: {action_text}")
        
        # ここでは簡単な実装として、テキストから数値を抽出
        # 実際の実装では、より高度な自然言語処理が必要
        try:
            # 数値の抽出（簡単な実装）
            import re
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", action_text)
            if len(numbers) >= 3:
                return {
                    "x": float(numbers[0]),
                    "y": float(numbers[1]),
                    "z": float(numbers[2])
                }
        except Exception as e:
            logger.error(f"行動の解析に失敗: {e}")
        
        logger.warning("有効な行動を抽出できませんでした。デフォルト値を使用します。")
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    
    def _execute_action(self, action: Dict[str, float]):
        """
        行動の実行
        
        Args:
            action: 実行する行動
        """
        try:
            # コントローラーを通じて行動を実行
            self.controller.execute_action(action)
            time.sleep(0.1)  # 実行時間のシミュレーション
        except Exception as e:
            logger.error(f"行動の実行に失敗: {e}")
    
    def _calculate_reward(self) -> float:
        """
        報酬の計算
        
        Returns:
            計算された報酬
        """
        # ここでは簡単な実装として、位置の安定性を報酬とする
        position = self.state["position"]
        reward = -np.sum(position ** 2)  # 位置が原点に近いほど報酬が高い
        return float(reward)
    
    def _update_model(self, reward: float):
        """
        モデルの更新
        
        Args:
            reward: 報酬
        """
        try:
            # 大脳皮質モデルの更新
            state_tensor = torch.tensor(self.state["joint_angles"], dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            
            # 損失の計算と逆伝播
            loss = self.cortical_model.compute_loss(state_tensor, reward_tensor)
            loss.backward()
            
            # オプティマイザのステップ
            self.cortical_model.optimizer.step()
            self.cortical_model.optimizer.zero_grad()
            
            logger.debug(f"モデルを更新しました（損失: {loss.item():.4f}）")
        except Exception as e:
            logger.error(f"モデルの更新に失敗: {e}")

def main():
    """メイン関数"""
    try:
        # APIキーの取得
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("警告: GOOGLE_API_KEY環境変数が設定されていません")
            api_key = input("Google AI Studio APIキーを入力してください: ")
        
        # デモの実行（可視化を有効化）
        demo = HumanoidDemo(api_key=api_key, visualize=True)
        demo.run_demo()
    except Exception as e:
        print(f"エラー: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 