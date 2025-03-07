import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import json
from pathlib import Path
import threading
import torch
# 内部モジュールのインポート
from ..cortical.neurotransmitters import NeurotransmitterSystem, GlutamateSystem, GABASystem, DrugSystem
from ..llm.langchain_integration import MultimodalLLMSystem, create_multimodal_llm
from ..genesis.robot_interface import GenesisRobotInterface, create_robot_interface

# BioKANのインポート
from ..cortical.biokan import BioKAN

# LLMエージェントのインポート
from ..core.multi_llm_agent import MultiLLMAgent

logger = logging.getLogger(__name__)

class IntegratedSystem:
    """
    神経伝達物質システム、LangChainマルチモーダルLLM、Genesisロボットインターフェースを
    統合した総合システム。生物学的に妥当なモデルでヒューマノイドロボットを制御します。
    
    BioKANを大脳皮質として三値入力（-1:抑制、0:中立、1:興奮）を処理し、
    LangChainを通じて複数のLLM（OpenAI、Claude、Gemini）と統合します。
    Genesisで物理演算を行い、リアル空間とのフィードバックを取得し、
    説明可能なヒューマノイドロボット制御を実現します。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        統合システムを初期化します
        
        Args:
            config_path: 設定ファイルパス
        """
        # 設定
        self.config = {
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
                'input_dim': 100,
                'hidden_dim': 256,
                'output_dim': 50,
                'num_layers': 6,
                'learning_rate': 0.001
            },
            'llm': {
                'default_provider': 'openai',  # openai, google, anthropic
                'alternative_providers': ['google', 'anthropic'],
                'fallback_enabled': True,
                'temperature': 0.7,
                'max_tokens': 1024
            },
            'robot_interface': {
                'simulation_mode': True
            }
        }
        
        # 設定ファイルからのロード
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # マージ
                    self._merge_config(loaded_config)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗しました: {str(e)}")
                
        # サブシステムの設定ファイルパス
        self.nt_config_path = None
        self.llm_config_path = None
        self.robot_config_path = None
        
        if config_path:
            config_dir = os.path.dirname(config_path)
            self.nt_config_path = os.path.join(config_dir, 'neurotransmitters_config.json')
            self.llm_config_path = os.path.join(config_dir, 'llm_config.json')
            self.robot_config_path = os.path.join(config_dir, 'robot_config.json')
            
        # 必要なディレクトリの作成
        self._create_directories()
        
        # サブシステムの初期化
        logger.info("神経伝達物質システムを初期化中...")
        self.nt_system = NeurotransmitterSystem()
        self.glutamate_system = GlutamateSystem()
        self.gaba_system = GABASystem()
        self.drug_system = DrugSystem(self.nt_system)
        
        # BioKAN大脳皮質モデルの初期化
        logger.info("BioKAN大脳皮質モデルを初期化中...")
        try:
            self.biokan = BioKAN(
                input_dim=self.config['cortical_model']['input_dim'],
                hidden_dim=self.config['cortical_model']['hidden_dim'],
                output_dim=self.config['cortical_model']['output_dim'],
                num_layers=self.config['cortical_model']['num_layers'],
                learning_rate=self.config['cortical_model']['learning_rate']
            )
            self.biokan_available = True
        except Exception as e:
            logger.error(f"BioKAN初期化エラー: {str(e)}")
            self.biokan_available = False
            self.biokan = None
        
        # LLMシステムの初期化
        logger.info("LLMシステムを初期化中...")
        self.llm_system = create_multimodal_llm(self.llm_config_path)
        
        # MultiLLMAgentの初期化（OpenAI、Claude、Geminiのいずれかを使用）
        try:
            self.llm_agent = MultiLLMAgent(
                provider=self.config['llm']['default_provider'],
                temperature=self.config['llm']['temperature'],
                max_tokens=self.config['llm']['max_tokens'],
                fallback_providers=self.config['llm']['alternative_providers'] if self.config['llm']['fallback_enabled'] else []
            )
            self.llm_agent_available = True
        except Exception as e:
            logger.error(f"MultiLLMAgent初期化エラー: {str(e)}")
            self.llm_agent_available = False
            self.llm_agent = None
        
        # ロボットインターフェースの初期化
        logger.info("Genesisロボットインターフェースを初期化中...")
        self.robot_interface = create_robot_interface(self.robot_config_path)
        
        # 内部状態
        self.running = False
        self.sensor_data = {}
        self.cortical_activation = {}
        self.motor_commands = {}
        self.last_explanation = {}
        
        # コルモゴロフアーノルドネットワーク（KAN）の三値入力処理用状態
        self.trinary_input_state = {
            "inhibitory": np.zeros(self.config['cortical_model']['input_dim']),  # -1: 抑制
            "neutral": np.ones(self.config['cortical_model']['input_dim']),      # 0: 中立
            "excitatory": np.zeros(self.config['cortical_model']['input_dim'])   # 1: 興奮
        }
        
        # 中枢神経用薬デフォルトレベルの設定
        for nt, level in self.config['neurotransmitters']['default_levels'].items():
            self.nt_system.set_level(nt, level)
            
        # 制御ループのスレッド
        self.control_thread = None
        
        logger.info("統合システムを初期化しました")
        
    def _merge_config(self, new_config: Dict) -> None:
        """設定を再帰的にマージします"""
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self._merge_config_recursive(self.config[key], value)
                else:
                    self.config[key] = value
                    
    def _merge_config_recursive(self, target: Dict, source: Dict) -> None:
        """設定を再帰的にマージします（内部ヘルパー）"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(target[key], value)
            else:
                target[key] = value
                
    def _create_directories(self) -> None:
        """必要なディレクトリを作成します"""
        directories = [
            self.config['system']['data_dir'],
            self.config['system']['knowledge_dir'],
            self.config['system']['log_dir'],
            self.config['system']['explanation_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def start(self) -> None:
        """システムの実行を開始します"""
        if self.running:
            logger.warning("システムは既に実行中です")
            return
            
        logger.info("システムを開始します")
        self.running = True
        
        # 制御ループのスレッドを開始
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
    def stop(self) -> None:
        """システムの実行を停止します"""
        if not self.running:
            logger.warning("システムは実行されていません")
            return
            
        logger.info("システムを停止します")
        self.running = False
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5.0)
            
    def _control_loop(self) -> None:
        """メイン制御ループ"""
        logger.info("制御ループを開始します")
        
        while self.running:
            try:
                # センサーデータの取得
                self.sensor_data = self.robot_interface.get_sensor_data()
                
                # 三値入力の生成
                trinary_input = self._generate_trinary_input(self.sensor_data)
                
                # BioKAN処理（大脳皮質モデル）
                if self.biokan_available:
                    # 三値入力を二値ベクトルに変換
                    biokan_input = self._convert_trinary_to_biokan_input(trinary_input)
                    cortical_output = self.biokan(torch.tensor(biokan_input, dtype=torch.float32))
                    self.cortical_activation = cortical_output.detach().numpy()
                else:
                    # フォールバック処理
                    self.cortical_activation = np.random.random(self.config['cortical_model']['output_dim']) * 2 - 1
                
                # 神経伝達物質レベルの更新
                self._update_neurotransmitter_levels()
                
                # LLMエージェントによる高次処理
                if self.llm_agent_available:
                    llm_input = {
                        "sensor_data": self.sensor_data,
                        "cortical_activation": self.cortical_activation.tolist(),
                        "neurotransmitter_levels": self.nt_system.get_all_levels(),
                        "goal": "現在の状況を分析し、適切な行動を選択してください"
                    }
                    
                    llm_response = self.llm_agent.process(llm_input)
                    
                    # 行動コマンドの生成
                    action = llm_response.get("action", {})
                    self.motor_commands = self._generate_motor_commands(action)
                    
                    # 説明の保存
                    self.last_explanation = {
                        "thought_process": llm_response.get("thought_process", ""),
                        "explanation": llm_response.get("explanation", ""),
                        "timestamp": time.time()
                    }
                else:
                    # LLMなしの場合は直接BioKANの出力からモーターコマンドを生成
                    self.motor_commands = self._direct_motor_mapping(self.cortical_activation)
                
                # ロボットへのコマンド送信
                self.robot_interface.send_commands(self.motor_commands)
                
                # フィードバック情報の収集と学習
                feedback = self.robot_interface.get_feedback()
                if self.biokan_available:
                    reward = self._calculate_reward(feedback)
                    self.biokan.update(torch.tensor([reward], dtype=torch.float32))
                
                # 少し待機
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"制御ループでエラーが発生しました: {str(e)}")
                time.sleep(1.0)  # エラー後の回復のために少し待機
    
    def _generate_trinary_input(self, sensor_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        センサーデータから三値入力（-1: 抑制、0: 中立、1: 興奮）を生成します
        
        Args:
            sensor_data: センサーデータ
            
        Returns:
            三値入力
        """
        # センサー値の正規化
        normalized_data = {}
        for key, value in sensor_data.items():
            if isinstance(value, (int, float)):
                normalized_data[key] = max(-1.0, min(1.0, value))
            elif isinstance(value, (list, np.ndarray)):
                normalized_data[key] = np.clip(np.array(value), -1.0, 1.0)
        
        # 三値入力の更新
        inhibitory = np.zeros(self.config['cortical_model']['input_dim'])
        neutral = np.zeros(self.config['cortical_model']['input_dim'])
        excitatory = np.zeros(self.config['cortical_model']['input_dim'])
        
        # センサーデータから三値マッピング
        # ここでは簡単な実装例: 値が-0.3未満なら抑制、0.3以上なら興奮、それ以外は中立
        for i, (key, value) in enumerate(normalized_data.items()):
            if isinstance(value, (int, float)):
                idx = i % self.config['cortical_model']['input_dim']
                if value < -0.3:
                    inhibitory[idx] = 1.0
                elif value > 0.3:
                    excitatory[idx] = 1.0
                else:
                    neutral[idx] = 1.0
            elif isinstance(value, (list, np.ndarray)):
                for j, v in enumerate(value):
                    idx = (i + j) % self.config['cortical_model']['input_dim']
                    if v < -0.3:
                        inhibitory[idx] = 1.0
                    elif v > 0.3:
                        excitatory[idx] = 1.0
                    else:
                        neutral[idx] = 1.0
        
        # 結果を保存
        self.trinary_input_state = {
            "inhibitory": inhibitory,
            "neutral": neutral,
            "excitatory": excitatory
        }
        
        return self.trinary_input_state
    
    def _convert_trinary_to_biokan_input(self, trinary_input: Dict[str, np.ndarray]) -> np.ndarray:
        """
        三値入力をBioKAN入力に変換します
        
        Args:
            trinary_input: 三値入力（抑制、中立、興奮）
            
        Returns:
            BioKAN入力ベクトル
        """
        # 三値入力を結合して一つのベクトルにする
        inhibitory = trinary_input["inhibitory"]
        neutral = trinary_input["neutral"]
        excitatory = trinary_input["excitatory"]
        
        # [-1, 0, 1]の値に変換
        biokan_input = np.zeros(self.config['cortical_model']['input_dim'])
        biokan_input -= inhibitory  # -1: 抑制
        biokan_input += excitatory  # +1: 興奮
        # neutral部分は0のまま
        
        return biokan_input
    
    def _update_neurotransmitter_levels(self) -> None:
        """BioKANと連動して神経伝達物質レベルを更新します"""
        if not self.biokan_available:
            return
            
        # BioKANから神経伝達物質レベルを取得
        nt_levels = self.biokan.get_neurotransmitter_levels()
        
        # 神経伝達物質システムを更新
        for nt, level in nt_levels.items():
            self.nt_system.set_level(nt, level)
    
    def _generate_motor_commands(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLMからの行動をモーターコマンドに変換します
        
        Args:
            action: LLMからの行動情報
            
        Returns:
            モーターコマンド
        """
        motor_commands = {}
        
        # 行動タイプに応じたモーターコマンドの生成
        action_type = action.get("type", "")
        parameters = action.get("parameters", {})
        
        if action_type == "move_forward":
            motor_commands["left_leg"] = 0.5
            motor_commands["right_leg"] = 0.5
        elif action_type == "turn_left":
            motor_commands["left_leg"] = -0.2
            motor_commands["right_leg"] = 0.5
        elif action_type == "turn_right":
            motor_commands["left_leg"] = 0.5
            motor_commands["right_leg"] = -0.2
        elif action_type == "stop":
            motor_commands["left_leg"] = 0.0
            motor_commands["right_leg"] = 0.0
        elif action_type == "greet":
            motor_commands["right_arm"] = 0.8
            motor_commands["head"] = 0.2
        elif action_type == "pick_up":
            motor_commands["right_arm"] = -0.5
            motor_commands["right_hand"] = 0.9
        else:
            # デフォルト: 静止状態
            motor_commands["left_leg"] = 0.0
            motor_commands["right_leg"] = 0.0
            motor_commands["left_arm"] = 0.0
            motor_commands["right_arm"] = 0.0
            motor_commands["head"] = 0.0
            
        return motor_commands
    
    def _direct_motor_mapping(self, cortical_output: np.ndarray) -> Dict[str, Any]:
        """
        BioKANの出力から直接モーターコマンドを生成します（LLMなしの場合）
        
        Args:
            cortical_output: 大脳皮質モデルの出力
            
        Returns:
            モーターコマンド
        """
        # 簡単なマッピングの例
        motor_map = {
            "head": 0,
            "left_arm": 1,
            "right_arm": 2,
            "left_hand": 3,
            "right_hand": 4,
            "torso": 5,
            "left_leg": 6,
            "right_leg": 7
        }
        
        motor_commands = {}
        for name, idx in motor_map.items():
            if idx < len(cortical_output):
                # -1.0～1.0の出力値をそのまま使用
                motor_commands[name] = float(cortical_output[idx])
        
        return motor_commands
    
    def _calculate_reward(self, feedback: Dict[str, Any]) -> float:
        """
        ロボットからのフィードバックを報酬に変換します
        
        Args:
            feedback: ロボットからのフィードバック
            
        Returns:
            報酬値 (-1.0 ～ 1.0)
        """
        # フィードバックに基づく報酬計算の例
        reward = 0.0
        
        # 転倒検出
        if feedback.get("fallen", False):
            reward -= 0.8
        
        # 目標達成
        if feedback.get("goal_reached", False):
            reward += 1.0
        
        # エネルギー効率
        energy = feedback.get("energy", 0.0)
        if energy < 0.3:  # 省エネの場合
            reward += 0.2
        
        # 速度に応じた報酬
        velocity = feedback.get("velocity", 0.0)
        if velocity > 0.5:  # 速い動きを奨励
            reward += 0.3
        
        return max(-1.0, min(1.0, reward))
    
    def get_explanation(self, query: Optional[str] = None) -> Dict[str, Any]:
        """
        システムの動作の説明を取得します
        
        Args:
            query: 問い合わせ文字列（オプション）
            
        Returns:
            説明情報
        """
        if not query:
            # 直近の説明を返す
            return self.last_explanation
        
        try:
            # 新しい説明の生成
            explanation = self._generate_explanation(query)
            self.last_explanation = explanation
            return explanation
        except Exception as e:
            logger.error(f"説明の生成中にエラーが発生しました: {str(e)}")
            return {
                "thought_process": "エラーが発生しました。",
                "explanation": f"説明の生成に失敗しました: {str(e)}",
                "timestamp": time.time()
            }
    
    def _generate_explanation(self, query: str) -> Dict[str, Any]:
        """
        クエリに応じた説明を生成します
        
        Args:
            query: 問い合わせ文字列
            
        Returns:
            説明情報
        """
        # LLMエージェントが利用可能な場合はそれを使用
        if self.llm_agent_available:
            llm_input = {
                "query": query,
                "sensor_data": self.sensor_data,
                "cortical_activation": self.cortical_activation.tolist() if isinstance(self.cortical_activation, np.ndarray) else self.cortical_activation,
                "neurotransmitter_levels": self.nt_system.get_all_levels(),
                "motor_commands": self.motor_commands,
                "goal": "ユーザーの質問に対して、脳科学的に正確で分かりやすい説明を提供してください"
            }
            
            llm_response = self.llm_agent.process(llm_input)
            
            return {
                "thought_process": llm_response.get("thought_process", ""),
                "explanation": llm_response.get("explanation", ""),
                "timestamp": time.time()
            }
        
        # LLMシステムが利用可能な場合はそれをフォールバックとして使用
        if self.llm_system:
            # 説明用プロンプトの構築
            prompt = f"""
            質問: {query}
            
            現在のシステム状態:
            - センサーデータ: {json.dumps(self.sensor_data, indent=2)}
            - 大脳皮質活性化状態: {self.cortical_activation.tolist() if isinstance(self.cortical_activation, np.ndarray) else self.cortical_activation}
            - 神経伝達物質レベル: {json.dumps(self.nt_system.get_all_levels(), indent=2)}
            - モーターコマンド: {json.dumps(self.motor_commands, indent=2)}
            
            以上の情報を基に、質問に対して脳科学的に正確で分かりやすい説明を提供してください。
            """
            
            # システムメッセージの設定
            system_message = """
            あなたは高度な神経科学とAIの専門家として、生物学的に妥当なヒューマノイドロボット制御システムについて説明します。
            質問に対して、神経伝達物質、受容体、薬物効果、皮質モデル、感覚運動統合に焦点を当てて、
            科学的に正確かつ分かりやすく回答してください。
            専門用語を適切に使用しつつ、一般の人にも理解できるよう努めてください。
            """
            
            # LLMから応答を生成
            response = self.llm_system.generate_response(prompt, system_message)
            return {
                "thought_process": "LLMシステムによる直接応答",
                "explanation": response,
                "timestamp": time.time()
            }
        
        # どちらも利用できない場合は基本的な説明を返す
        return {
            "thought_process": "基本的な自動応答",
            "explanation": f"質問: {query}\n\n現在のシステムの状態に基づく説明機能は現在利用できません。",
            "timestamp": time.time()
        }
    
    def get_natural_language_explanation(self, query: str) -> str:
        """
        自然言語での説明を取得します
        
        Args:
            query: 問い合わせ文字列
            
        Returns:
            説明文字列
        """
        explanation_data = self.get_explanation(query)
        return explanation_data.get("explanation", "説明を生成できませんでした。")

def create_integrated_system(config_path: Optional[str] = None) -> IntegratedSystem:
    """
    統合システムのファクトリー関数
    
    Args:
        config_path: 設定ファイルパス
        
    Returns:
        統合システムのインスタンス
    """
    return IntegratedSystem(config_path) 