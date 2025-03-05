"""
統合脳システム

BioKANを大脳皮質として、Gemini Proを高次脳機能として統合したシステム。
ヒューマノイドロボットの制御に使用します。
"""

import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import time

from ..cortical.biokan_cortex import CorticalBioKAN
from ..llm.gemini_agent import GeminiAgent
from ..utils.logging import get_logger

logger = get_logger(__name__)

class IntegratedBrain:
    """
    統合脳システム
    
    BioKANを大脳皮質として、Gemini Proを高次脳機能として統合したシステム。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cortical_config: Optional[Dict[str, Any]] = None,
        gemini_config: Optional[Dict[str, Any]] = None
    ):
        """
        初期化
        
        Args:
            api_key: Google AI Studio APIキー
            cortical_config: 大脳皮質モデルの設定
            gemini_config: Geminiモデルの設定
        """
        logger.info("統合脳システムを初期化しています...")
        
        # APIキーの確認
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            error_msg = "GOOGLE_API_KEY環境変数が設定されていません"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 大脳皮質モデルの設定
        self.cortical_config = cortical_config or {
            "input_dim": 100,
            "hidden_dim": 256,
            "output_dim": 50,
            "num_layers": 6,
            "learning_rate": 0.001
        }
        
        # Geminiモデルの設定
        self.gemini_config = gemini_config or {
            "model_name": "gemini-pro",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # 大脳皮質モデルの作成
        logger.info("大脳皮質モデル（BioKAN）を作成しています...")
        self.cortical_model = CorticalBioKAN(**self.cortical_config)
        
        # Geminiエージェントの初期化
        logger.info("Geminiエージェントを初期化しています...")
        self.gemini_agent = GeminiAgent(
            api_key=self.api_key,
            **self.gemini_config
        )
        
        # 内部状態
        self.internal_state = {
            "cortical_activity": np.zeros(self.cortical_config["output_dim"]),
            "neurotransmitter_levels": self.cortical_model.get_neurotransmitter_levels(),
            "emotional_state": {
                "valence": 0.0,  # 感情の正負（-1.0〜1.0）
                "arousal": 0.5,  # 覚醒度（0.0〜1.0）
                "dominance": 0.5  # 支配性（0.0〜1.0）
            },
            "memory": {
                "short_term": [],
                "working": {},
                "episodic": []
            },
            "attention": {
                "focus": None,
                "level": 0.5
            }
        }
        
        # 学習パラメータ
        self.learning_params = {
            "learning_rate": 0.001,
            "discount_factor": 0.95,
            "exploration_rate": 0.1
        }
        
        logger.info("統合脳システムの初期化が完了しました")
    
    def process_input(
        self,
        sensory_input: Dict[str, Any],
        reward: float = 0.0
    ) -> Dict[str, Any]:
        """
        入力の処理
        
        Args:
            sensory_input: 感覚入力
            reward: 報酬値
            
        Returns:
            処理結果と行動
        """
        logger.info("入力を処理しています...")
        
        # 入力の前処理
        processed_input = self._preprocess_input(sensory_input)
        
        # 大脳皮質モデルによる処理
        cortical_output = self._process_cortical(processed_input, reward)
        
        # Geminiエージェントによる高次処理
        gemini_context = {
            "sensory_input": sensory_input,
            "cortical_output": cortical_output,
            "internal_state": self.internal_state,
            "reward": reward
        }
        
        # 行動決定のためのプロンプト
        prompt = self._generate_prompt(gemini_context)
        
        # Geminiエージェントからの応答
        logger.info("Geminiエージェントに問い合わせています...")
        gemini_response = self.gemini_agent.process_input(prompt, gemini_context)
        
        # 応答の解析と行動の抽出
        action = self._parse_gemini_response(gemini_response)
        
        # 内部状態の更新
        self._update_internal_state(sensory_input, cortical_output, action, reward)
        
        # 結果の返却
        result = {
            "action": action,
            "cortical_output": cortical_output,
            "internal_state": self.internal_state,
            "gemini_response": gemini_response
        }
        
        return result
    
    def _preprocess_input(self, sensory_input: Dict[str, Any]) -> torch.Tensor:
        """
        入力の前処理
        
        Args:
            sensory_input: 感覚入力
            
        Returns:
            処理された入力テンソル
        """
        # 入力の正規化と変換
        # 実際の実装では、様々な感覚入力を統一的なテンソル形式に変換
        
        # 簡単な実装として、位置と関節角度を結合
        position = sensory_input.get("position", np.zeros(3))
        joint_angles = sensory_input.get("joint_angles", np.zeros(20))
        
        # 入力ベクトルの作成
        input_vector = np.concatenate([position, joint_angles])
        
        # 入力次元の調整
        if len(input_vector) < self.cortical_config["input_dim"]:
            padding = np.zeros(self.cortical_config["input_dim"] - len(input_vector))
            input_vector = np.concatenate([input_vector, padding])
        elif len(input_vector) > self.cortical_config["input_dim"]:
            input_vector = input_vector[:self.cortical_config["input_dim"]]
        
        # テンソルに変換
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        
        return input_tensor
    
    def _process_cortical(
        self,
        input_tensor: torch.Tensor,
        reward: float
    ) -> Dict[str, Any]:
        """
        大脳皮質モデルによる処理
        
        Args:
            input_tensor: 入力テンソル
            reward: 報酬値
            
        Returns:
            大脳皮質の出力
        """
        # 報酬テンソルの作成
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        # 大脳皮質モデルの順伝播
        with torch.no_grad():
            output_tensor = self.cortical_model.forward(input_tensor)
        
        # 学習（報酬がある場合）
        if abs(reward) > 0.01:
            loss = self.cortical_model.update(input_tensor, reward_tensor)
            logger.debug(f"大脳皮質モデルを更新しました（損失: {loss:.4f}）")
        
        # 出力の変換
        output_np = output_tensor.detach().numpy()
        
        # 神経伝達物質レベルの取得
        neurotransmitter_levels = self.cortical_model.get_neurotransmitter_levels()
        
        # 出力の整形
        cortical_output = {
            "activity": output_np,
            "neurotransmitter_levels": neurotransmitter_levels
        }
        
        return cortical_output
    
    def _generate_prompt(self, context: Dict[str, Any]) -> str:
        """
        Geminiエージェント用のプロンプト生成
        
        Args:
            context: コンテキスト情報
            
        Returns:
            生成されたプロンプト
        """
        # 感覚入力の文字列化
        sensory_str = "\n".join([
            f"{key}: {value}" for key, value in context["sensory_input"].items()
        ])
        
        # 神経伝達物質レベルの文字列化
        nt_levels = context["cortical_output"]["neurotransmitter_levels"]
        nt_str = "\n".join([
            f"{key}: {value:.2f}" for key, value in nt_levels.items()
        ])
        
        # 感情状態の文字列化
        emotion = context["internal_state"]["emotional_state"]
        emotion_str = "\n".join([
            f"{key}: {value:.2f}" for key, value in emotion.items()
        ])
        
        # プロンプトの構築
        prompt = f"""あなたはヒューマノイドロボットの脳として機能しています。
現在の状態に基づいて、次の行動を決定してください。

## 感覚入力
{sensory_str}

## 大脳皮質の活動
活動レベル: {np.mean(context["cortical_output"]["activity"]):.2f}

## 神経伝達物質レベル
{nt_str}

## 感情状態
{emotion_str}

## 報酬
{context["reward"]:.2f}

以上の情報に基づいて、次の行動を決定してください。
行動は以下の形式で返してください：

行動:
x: [x座標]
y: [y座標]
z: [z座標]
理由: [行動の理由]
"""
        
        return prompt
    
    def _parse_gemini_response(self, response: str) -> Dict[str, Any]:
        """
        Geminiエージェントの応答を解析
        
        Args:
            response: Geminiエージェントの応答
            
        Returns:
            解析された行動
        """
        logger.debug(f"Geminiの応答: {response}")
        
        # デフォルトの行動
        action = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "reason": "デフォルト行動"
        }
        
        try:
            # 行動の抽出
            import re
            
            # x座標の抽出
            x_match = re.search(r"x:\s*([-+]?\d*\.\d+|\d+)", response)
            if x_match:
                action["x"] = float(x_match.group(1))
            
            # y座標の抽出
            y_match = re.search(r"y:\s*([-+]?\d*\.\d+|\d+)", response)
            if y_match:
                action["y"] = float(y_match.group(1))
            
            # z座標の抽出
            z_match = re.search(r"z:\s*([-+]?\d*\.\d+|\d+)", response)
            if z_match:
                action["z"] = float(z_match.group(1))
            
            # 理由の抽出
            reason_match = re.search(r"理由:\s*(.+?)(?:\n|$)", response)
            if reason_match:
                action["reason"] = reason_match.group(1).strip()
            
        except Exception as e:
            logger.error(f"Gemini応答の解析に失敗: {e}")
        
        return action
    
    def _update_internal_state(
        self,
        sensory_input: Dict[str, Any],
        cortical_output: Dict[str, Any],
        action: Dict[str, Any],
        reward: float
    ):
        """
        内部状態の更新
        
        Args:
            sensory_input: 感覚入力
            cortical_output: 大脳皮質の出力
            action: 決定された行動
            reward: 報酬値
        """
        # 大脳皮質活動の更新
        self.internal_state["cortical_activity"] = cortical_output["activity"]
        
        # 神経伝達物質レベルの更新
        self.internal_state["neurotransmitter_levels"] = cortical_output["neurotransmitter_levels"]
        
        # 感情状態の更新
        nt_levels = cortical_output["neurotransmitter_levels"]
        
        # ドーパミンとセロトニンに基づく感情価（valence）の更新
        valence = (nt_levels["dopamine"] - 0.5) * 2.0  # -1.0〜1.0
        valence += (nt_levels["serotonin"] - 0.5) * 1.0  # セロトニンの影響は小さめ
        valence = max(-1.0, min(1.0, valence))  # -1.0〜1.0に制限
        
        # ノルアドレナリンとアセチルコリンに基づく覚醒度（arousal）の更新
        arousal = nt_levels["noradrenaline"]
        arousal += (nt_levels["acetylcholine"] - 0.5) * 0.5  # アセチルコリンの影響は小さめ
        arousal = max(0.0, min(1.0, arousal))  # 0.0〜1.0に制限
        
        # グルタミン酸とGABAに基づく支配性（dominance）の更新
        dominance = nt_levels["glutamate"]
        dominance -= (nt_levels["gaba"] - 0.5) * 1.0  # GABAは抑制的
        dominance = max(0.0, min(1.0, dominance))  # 0.0〜1.0に制限
        
        self.internal_state["emotional_state"] = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance
        }
        
        # 短期記憶の更新
        memory_entry = {
            "timestamp": time.time(),
            "sensory_input": sensory_input,
            "action": action,
            "reward": reward
        }
        
        self.internal_state["memory"]["short_term"].append(memory_entry)
        
        # 短期記憶のサイズ制限
        max_short_term_size = 10
        if len(self.internal_state["memory"]["short_term"]) > max_short_term_size:
            # 古い記憶を削除
            self.internal_state["memory"]["short_term"] = self.internal_state["memory"]["short_term"][-max_short_term_size:]
            
            # 重要な記憶はエピソード記憶に移動
            if reward > 0.5 or reward < -0.5:
                self.internal_state["memory"]["episodic"].append(memory_entry)
        
        # 注意の更新
        if "focus" in sensory_input:
            self.internal_state["attention"]["focus"] = sensory_input["focus"]
        
        # 注意レベルの更新（アセチルコリンとノルアドレナリンに基づく）
        attention_level = (nt_levels["acetylcholine"] + nt_levels["noradrenaline"]) / 2.0
        self.internal_state["attention"]["level"] = attention_level
    
    def simulate_drug_effect(self, drug_name: str, dose: float = 1.0) -> Dict[str, float]:
        """
        薬物効果のシミュレーション
        
        Args:
            drug_name: 薬物名
            dose: 投与量（0-1）
            
        Returns:
            変更された神経伝達物質レベル
        """
        return self.cortical_model.simulate_drug_effect(drug_name, dose)
    
    def get_emotional_state(self) -> Dict[str, float]:
        """
        感情状態の取得
        
        Returns:
            感情状態
        """
        return self.internal_state["emotional_state"]
    
    def get_neurotransmitter_levels(self) -> Dict[str, float]:
        """
        神経伝達物質レベルの取得
        
        Returns:
            神経伝達物質レベル
        """
        return self.internal_state["neurotransmitter_levels"]
    
    def get_neurotransmitter_effects(self) -> Dict[str, str]:
        """
        神経伝達物質の効果説明の取得
        
        Returns:
            各神経伝達物質の効果説明
        """
        return self.cortical_model.get_neurotransmitter_effects() 