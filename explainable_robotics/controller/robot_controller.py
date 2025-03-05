"""ロボットコントローラー

BioKANモデル、Geminiエージェント、Genesisビジュアライザーを統合する
ヒューマノイドロボット制御のためのメインコントローラーモジュール。
"""

import os
import time
import threading
import queue
import json
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

import torch
import numpy as np

# ロボットの主要コンポーネント
from ..cortical.biokan import BioKAN

# ユーティリティのインポート
from ..utils.logging import get_logger

# GeminiAgentのインポートを削除し、MultiLLMAgentをインポート
try:
    from explainable_robotics.core.multi_llm_agent import MultiLLMAgent
    HAVE_LLM_AGENT = True
except ImportError:
    logger.warning("MultiLLMAgentをインポートできませんでした。AI機能が制限されます。")
    HAVE_LLM_AGENT = False

# ロガーの設定
logger = get_logger(__name__)


class RobotController:
    """
    ヒューマノイドロボットコントローラー
    
    BioKANモデル（大脳皮質）、MultiLLMAgent（言語理解と行動生成）、
    GenesisVisualizer（可視化と物理シミュレーション）を統合して、
    ヒューマノイドロボットを制御します。
    """
    
    def __init__(
        self,
        robot_name: str = "Explainable Robot",
        model_path: Optional[str] = None,
        biokan_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        visualizer_config: Optional[Dict[str, Any]] = None,
        log_actions: bool = True,
        action_log_path: Optional[str] = None,
        safety_constraints: Optional[Dict[str, Any]] = None
    ):
        """
        初期化
        
        Args:
            robot_name: ロボットの名前
            model_path: ロボットモデルのパス
            biokan_config: BioKANモデルの設定
            llm_config: MultiLLMAgentの設定
            visualizer_config: ビジュアライザーの設定
            log_actions: 行動をログに記録するかどうか
            action_log_path: 行動ログのパス
            safety_constraints: 安全制約の設定
        """
        self.robot_name = robot_name
        self.model_path = model_path
        self.log_actions = log_actions
        
        # 現在の状態
        self.state = {
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "joint_angles": [],
            "sensor_data": {},
            "last_action": None,
            "last_reward": 0.0,
            "goals": [],
            "status": "initialized"
        }
        
        # 安全制約の設定
        self.safety_constraints = safety_constraints or {
            "max_velocity": 1.0,  # m/s
            "max_rotation_velocity": 1.0,  # rad/s
            "max_force": 50.0,  # N
            "min_obstacle_distance": 0.5,  # m
            "emergency_stop_enabled": True
        }
        
        # 行動キュー
        self.action_queue = queue.Queue()
        
        # 行動ログの設定
        self.action_log_path = action_log_path
        if log_actions and action_log_path is None:
            self.action_log_path = f"logs/actions_{int(time.time())}.jsonl"
            os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
        
        # デフォルト設定
        default_biokan_config = {
            "input_dim": 128,
            "hidden_dim": 256,
            "output_dim": 64,
            "num_layers": 6,
            "dropout": 0.1,
            "learning_rate": 1e-3
        }
        
        # llm_configのデフォルト値設定
        llm_config = llm_config or {}
        self.llm_config = {
            "provider": llm_config.get("provider", "gemini"),
            "api_key": llm_config.get("api_key", ""),
            "model_name": llm_config.get("model_name", None),
            "temperature": llm_config.get("temperature", 0.7),
            "max_output_tokens": llm_config.get("max_output_tokens", 1024),
            "safety_settings": llm_config.get("safety_settings", None),
            "use_memory": llm_config.get("use_memory", True),
            "memory_size": llm_config.get("memory_size", 10),
            "memory_path": llm_config.get("memory_path", "data/memory.json"),
            "local_model_path": llm_config.get("local_model_path", None),
            "project_id": llm_config.get("project_id", None),
            "location": llm_config.get("location", "us-central1"),
            "credentials_path": llm_config.get("credentials_path", None),
        }
        
        # 設定のマージ
        self.biokan_config = {**default_biokan_config, **(biokan_config or {})}
        
        # コンポーネントの初期化
        self._initialize_components()
        
        # スレッド初期化
        self.is_running = False
        self.control_thread = None
        
        logger.info(f"ロボットコントローラーを初期化しました（名前: {robot_name}）")
    
    def _initialize_components(self):
        """コンポーネントの初期化"""
        success = True
        
        # BioKANモデルの初期化
        logger.info("BioKANモデルを初期化中...")
        try:
            self.biokan = BioKAN(
                input_dim=self.biokan_config["input_dim"],
                hidden_dim=self.biokan_config["hidden_dim"],
                output_dim=self.biokan_config["output_dim"],
                num_layers=self.biokan_config["num_layers"],
                dropout=self.biokan_config["dropout"],
                learning_rate=self.biokan_config["learning_rate"]
            )
            logger.info("BioKANモデルの初期化が完了しました")
        except Exception as e:
            logger.error(f"BioKANモデルの初期化に失敗しました: {e}")
            self.biokan = None
            success = False
        
        # LLMエージェントの初期化
        if HAVE_LLM_AGENT:
            try:
                logger.info(f"{self.llm_config['provider']}ベースのLLMエージェントを初期化しています...")
                self.agent = MultiLLMAgent(
                    provider=self.llm_config["provider"],
                    api_key=self.llm_config["api_key"],
                    model_name=self.llm_config["model_name"],
                    temperature=self.llm_config["temperature"],
                    max_output_tokens=self.llm_config["max_output_tokens"],
                    safety_settings=self.llm_config["safety_settings"],
                    use_memory=self.llm_config["use_memory"],
                    memory_size=self.llm_config["memory_size"],
                    memory_path=self.llm_config["memory_path"],
                    local_model_path=self.llm_config["local_model_path"],
                    project_id=self.llm_config["project_id"],
                    location=self.llm_config["location"],
                    credentials_path=self.llm_config["credentials_path"],
                )
                logger.info("LLMエージェントの初期化が完了しました。")
            except Exception as e:
                logger.error(f"LLMエージェントの初期化に失敗しました: {str(e)}")
                success = False
        else:
            logger.warning("LLMエージェントの初期化をスキップします。")
            self.agent = None
        
        # ビジュアライザーの初期化
        logger.info("Genesisビジュアライザーを初期化中...")
        try:
            self.visualizer = GenesisVisualizer(
                model_path=self.model_path,
                window_title=self.visualizer_config["window_title"],
                width=self.visualizer_config["width"],
                height=self.visualizer_config["height"],
                show_ui=self.visualizer_config["show_ui"],
                physics_enabled=self.visualizer_config["physics_enabled"]
            )
            logger.info("Genesisビジュアライザーの初期化が完了しました")
        except Exception as e:
            logger.error(f"Genesisビジュアライザーの初期化に失敗しました: {e}")
            self.visualizer = None
        
        return success
    
    def start(self):
        """ロボットの起動"""
        if self.is_running:
            logger.warning("ロボットは既に実行中です")
            return
        
        # ビジュアライザーの起動
        if self.visualizer:
            self.visualizer.start()
        
        # 制御ループスレッドの開始
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # 状態の更新
        self.state["status"] = "running"
        
        logger.info("ロボットコントローラーを起動しました")
    
    def stop(self):
        """ロボットの停止"""
        if not self.is_running:
            return
        
        # 制御ループの停止
        self.is_running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        # ビジュアライザーの停止
        if self.visualizer:
            self.visualizer.stop()
        
        # 状態の更新
        self.state["status"] = "stopped"
        
        logger.info("ロボットコントローラーを停止しました")
    
    def set_goal(self, goal: Union[str, Dict[str, Any]]):
        """
        目標の設定
        
        Args:
            goal: 自然言語または辞書形式の目標
        """
        if isinstance(goal, str):
            goal_obj = {"type": "natural_language", "description": goal}
        else:
            goal_obj = goal
        
        # 目標の追加
        self.state["goals"].append(goal_obj)
        
        # 優先度が指定されていない場合はデフォルト値を設定
        if "priority" not in goal_obj:
            goal_obj["priority"] = 1.0
        
        # タイムスタンプの追加
        goal_obj["timestamp"] = time.time()
        
        logger.info(f"目標を設定しました: {goal_obj}")
    
    def clear_goals(self):
        """全ての目標をクリア"""
        self.state["goals"] = []
        logger.info("全ての目標をクリアしました")
    
    def update_sensor_data(self, sensor_data: Dict[str, Any]):
        """
        センサーデータの更新
        
        Args:
            sensor_data: センサーデータ
        """
        # センサーデータのマージ
        self.state["sensor_data"].update(sensor_data)
        
        # センサーデータのタイムスタンプ更新
        self.state["sensor_data"]["timestamp"] = time.time()
    
    def execute_action(self, action: Dict[str, Any]) -> float:
        """
        行動の実行
        
        Args:
            action: 実行する行動
        
        Returns:
            報酬値
        """
        # 安全チェック
        if not self._check_safety(action):
            logger.warning(f"安全制約に違反する行動が拒否されました: {action}")
            return -1.0
        
        # 行動の種類に基づいて処理
        action_type = action.get("type", "unknown")
        parameters = action.get("parameters", {})
        reward = 0.0
        
        try:
            # 行動の実行
            if action_type == "move":
                reward = self._execute_move_action(parameters)
            elif action_type == "rotate":
                reward = self._execute_rotate_action(parameters)
            elif action_type == "joint_control":
                reward = self._execute_joint_control_action(parameters)
            elif action_type == "speak":
                reward = self._execute_speak_action(parameters)
            elif action_type == "grasp":
                reward = self._execute_grasp_action(parameters)
            elif action_type == "release":
                reward = self._execute_release_action(parameters)
            elif action_type == "composite":
                reward = self._execute_composite_action(parameters)
            else:
                logger.warning(f"未知の行動タイプ: {action_type}")
                reward = 0.0
            
            # 行動をログに記録
            if self.log_actions and self.action_log_path:
                self._log_action(action, reward)
            
            # 最後の行動と報酬を記録
            self.state["last_action"] = action
            self.state["last_reward"] = reward
            
            return reward
            
        except Exception as e:
            logger.error(f"行動の実行中にエラーが発生しました: {e}")
            return -1.0
    
    def queue_action(self, action: Dict[str, Any]):
        """
        行動のキューイング
        
        Args:
            action: キューに追加する行動
        """
        self.action_queue.put(action)
        logger.debug(f"行動をキューに追加しました: {action}")
    
    def save_state(self, filepath: str):
        """
        状態の保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        try:
            # BioKANモデルの神経伝達物質レベルを取得
            if self.biokan:
                nt_levels = self.biokan.get_neurotransmitter_levels()
            else:
                nt_levels = {}
            
            # 保存するデータの準備
            save_data = {
                "robot_name": self.robot_name,
                "timestamp": time.time(),
                "state": self.state,
                "neurotransmitter_levels": nt_levels,
                "safety_constraints": self.safety_constraints
            }
            
            # ディレクトリの作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # JSONとして保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"状態を保存しました: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"状態の保存中にエラーが発生しました: {e}")
            return False
    
    def load_state(self, filepath: str):
        """
        状態の読み込み
        
        Args:
            filepath: 読み込むファイルパス
        """
        try:
            # JSONから読み込み
            with open(filepath, 'r', encoding='utf-8') as f:
                load_data = json.load(f)
            
            # 状態の復元
            self.robot_name = load_data.get("robot_name", self.robot_name)
            self.state = load_data.get("state", self.state)
            self.safety_constraints = load_data.get("safety_constraints", self.safety_constraints)
            
            # BioKANモデルの神経伝達物質レベルを設定
            if self.biokan and "neurotransmitter_levels" in load_data:
                self.biokan.set_neurotransmitter_levels(load_data["neurotransmitter_levels"])
            
            logger.info(f"状態を読み込みました: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"状態の読み込み中にエラーが発生しました: {e}")
            return False
    
    def _control_loop(self):
        """制御ループ"""
        logger.info("制御ループを開始しました")
        
        while self.is_running:
            try:
                # 行動キューからの処理
                try:
                    action = self.action_queue.get(timeout=0.1)
                    reward = self.execute_action(action)
                    self.action_queue.task_done()
                    
                    # BioKANモデルの更新
                    if self.biokan:
                        self.biokan.update(torch.tensor([reward]))
                except queue.Empty:
                    pass
                
                # 現在の目標がある場合、自律的に行動を決定
                current_goals = self.state["goals"]
                if current_goals and self.agent and self.biokan:
                    # 脳の状態（神経伝達物質レベル）を取得
                    nt_levels = self.biokan.get_neurotransmitter_levels()
                    
                    # 入力データの準備
                    input_data = {
                        "state": {
                            "position": self.state["position"],
                            "rotation": self.state["rotation"],
                            "joint_angles": self.state["joint_angles"]
                        },
                        "sensor_data": self.state["sensor_data"],
                        "goals": current_goals,
                    }
                    
                    # コンテキストの準備
                    context = {
                        "robot_name": self.robot_name,
                        "last_action": self.state["last_action"],
                        "last_reward": self.state["last_reward"],
                        "safety_constraints": self.safety_constraints
                    }
                    
                    # エージェントに判断を要求
                    response = self.agent.process(input_data, context, nt_levels)
                    
                    # レスポンスから行動を抽出
                    if "action" in response:
                        action = response["action"]
                        confidence = action.get("confidence", 0.0)
                        
                        # 信頼度が低い場合は警告ログを出力
                        if confidence < 0.5:
                            logger.warning(f"低信頼度の行動: {action} (信頼度: {confidence})")
                        
                        # 行動を実行
                        reward = self.execute_action(action)
                        
                        # BioKANモデルの更新
                        if self.biokan:
                            self.biokan.update(torch.tensor([reward]))
                
                # ビジュアライザーの更新
                if self.visualizer:
                    # ポーズの更新
                    pose_data = {
                        "position": self.state["position"],
                        "rotation": self.state["rotation"],
                        "joint_angles": self.state["joint_angles"]
                    }
                    self.visualizer.queue_pose_update(pose_data)
                    
                    # 脳状態の更新
                    if self.biokan:
                        nt_levels = self.biokan.get_neurotransmitter_levels()
                        brain_data = {
                            "neurotransmitter_levels": nt_levels,
                            "emotional_state": self._derive_emotional_state(nt_levels)
                        }
                        self.visualizer.queue_brain_state_update(brain_data)
                
                # 短い待機
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"制御ループでエラーが発生しました: {e}")
                time.sleep(1.0)
    
    def _check_safety(self, action: Dict[str, Any]) -> bool:
        """
        安全制約のチェック
        
        Args:
            action: チェックする行動
        
        Returns:
            安全かどうか
        """
        # 非常停止が有効な場合はすべての移動行動を拒否
        if self.safety_constraints.get("emergency_stop_enabled", False):
            if action.get("type") in ["move", "rotate", "joint_control"]:
                return False
        
        # 行動の種類に基づいてチェック
        action_type = action.get("type", "unknown")
        parameters = action.get("parameters", {})
        
        if action_type == "move":
            # 速度制約のチェック
            velocity = parameters.get("velocity", 0.0)
            max_velocity = self.safety_constraints.get("max_velocity", 1.0)
            if abs(velocity) > max_velocity:
                return False
            
            # 障害物距離のチェック
            direction = parameters.get("direction", [0, 0, 0])
            min_distance = self._get_obstacle_distance(direction)
            if min_distance < self.safety_constraints.get("min_obstacle_distance", 0.5):
                return False
        
        elif action_type == "rotate":
            # 回転速度制約のチェック
            velocity = parameters.get("velocity", 0.0)
            max_velocity = self.safety_constraints.get("max_rotation_velocity", 1.0)
            if abs(velocity) > max_velocity:
                return False
        
        elif action_type == "joint_control":
            # 関節力制約のチェック
            force = parameters.get("force", 0.0)
            max_force = self.safety_constraints.get("max_force", 50.0)
            if abs(force) > max_force:
                return False
        
        return True
    
    def _get_obstacle_distance(self, direction: List[float]) -> float:
        """
        障害物までの距離を取得
        
        Args:
            direction: 方向ベクトル
        
        Returns:
            障害物までの最小距離
        """
        # センサーデータから障害物情報を取得
        obstacles = self.state["sensor_data"].get("obstacles", [])
        
        # 障害物がない場合は大きな値を返す
        if not obstacles:
            return 100.0
        
        # 障害物までの最小距離を計算
        min_distance = float('inf')
        for obstacle in obstacles:
            distance = obstacle.get("distance", float('inf'))
            if distance < min_distance:
                # 方向が一致する場合のみ考慮
                obstacle_direction = obstacle.get("direction", [0, 0, 0])
                dot_product = sum(a * b for a, b in zip(direction, obstacle_direction))
                if dot_product > 0:  # 同じ方向の場合
                    min_distance = distance
        
        return min_distance if min_distance != float('inf') else 100.0
    
    def _derive_emotional_state(self, nt_levels: Dict[str, float]) -> Dict[str, Any]:
        """
        神経伝達物質レベルから感情状態を導出
        
        Args:
            nt_levels: 神経伝達物質レベル
        
        Returns:
            感情状態
        """
        # 主要な感情と強度のマッピング
        emotions = {
            "happy": nt_levels.get("dopamine", 0.5) * 0.7 + nt_levels.get("serotonin", 0.5) * 0.3,
            "sad": (1.0 - nt_levels.get("serotonin", 0.5)) * 0.8,
            "excited": nt_levels.get("dopamine", 0.5) * 0.5 + nt_levels.get("norepinephrine", 0.5) * 0.5,
            "anxious": nt_levels.get("norepinephrine", 0.5) * 0.7 + (1.0 - nt_levels.get("gaba", 0.5)) * 0.3,
            "calm": nt_levels.get("gaba", 0.5) * 0.6 + nt_levels.get("serotonin", 0.5) * 0.4,
            "focused": nt_levels.get("acetylcholine", 0.5) * 0.8 + nt_levels.get("norepinephrine", 0.5) * 0.2,
            "curious": nt_levels.get("dopamine", 0.5) * 0.4 + nt_levels.get("acetylcholine", 0.5) * 0.6
        }
        
        # 最大の感情を見つける
        primary_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            "primary_emotion": primary_emotion[0],
            "intensity": primary_emotion[1],
            "all_emotions": emotions
        }
    
    def _log_action(self, action: Dict[str, Any], reward: float):
        """
        行動のログ記録
        
        Args:
            action: 実行された行動
            reward: 報酬値
        """
        if not self.action_log_path:
            return
        
        # ログエントリの作成
        log_entry = {
            "timestamp": time.time(),
            "action": action,
            "reward": reward,
            "position": self.state["position"],
            "rotation": self.state["rotation"],
            "goals": self.state["goals"]
        }
        
        # JSONLとして追記
        with open(self.action_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def _execute_move_action(self, parameters: Dict[str, Any]) -> float:
        """
        移動行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        direction = parameters.get("direction", [0, 0, 0])
        distance = parameters.get("distance", 0.0)
        velocity = parameters.get("velocity", 1.0)
        
        # 方向ベクトルの正規化
        magnitude = sum(d ** 2 for d in direction) ** 0.5
        if magnitude > 0:
            normalized_direction = [d / magnitude for d in direction]
        else:
            normalized_direction = [0, 0, 0]
        
        # 新しい位置の計算
        new_position = [
            self.state["position"][0] + normalized_direction[0] * distance,
            self.state["position"][1] + normalized_direction[1] * distance,
            self.state["position"][2] + normalized_direction[2] * distance
        ]
        
        # 位置の更新
        self.state["position"] = new_position
        
        # 目標達成度に基づく報酬計算
        reward = 0.5  # デフォルトの報酬
        
        # 目標位置が指定されている場合は距離に基づいて報酬を調整
        for goal in self.state["goals"]:
            if goal.get("type") == "position" and "position" in goal:
                target_pos = goal["position"]
                current_pos = self.state["position"]
                
                # 以前の距離
                old_distance = sum((a - b) ** 2 for a, b in zip(self.state["position"], target_pos)) ** 0.5
                
                # 新しい距離
                new_distance = sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)) ** 0.5
                
                # 距離の減少に基づく報酬
                if new_distance < old_distance:
                    reward += 0.3
                else:
                    reward -= 0.1
        
        return max(0.0, min(1.0, reward))
    
    def _execute_rotate_action(self, parameters: Dict[str, Any]) -> float:
        """
        回転行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        axis = parameters.get("axis", [0, 1, 0])  # デフォルトはY軸
        angle = parameters.get("angle", 0.0)  # ラジアン
        velocity = parameters.get("velocity", 1.0)
        
        # 回転の適用
        if axis[0] > 0.5:  # X軸
            self.state["rotation"][0] += angle
        elif axis[1] > 0.5:  # Y軸
            self.state["rotation"][1] += angle
        elif axis[2] > 0.5:  # Z軸
            self.state["rotation"][2] += angle
        
        # 角度の正規化（-π から π の範囲に）
        for i in range(3):
            while self.state["rotation"][i] > 3.14159:
                self.state["rotation"][i] -= 2 * 3.14159
            while self.state["rotation"][i] < -3.14159:
                self.state["rotation"][i] += 2 * 3.14159
        
        # デフォルトの報酬
        reward = 0.4
        
        # 目標回転が指定されている場合は角度に基づいて報酬を調整
        for goal in self.state["goals"]:
            if goal.get("type") == "rotation" and "rotation" in goal:
                target_rot = goal["rotation"]
                current_rot = self.state["rotation"]
                
                # 角度差の計算
                angle_diff = sum((a - b) ** 2 for a, b in zip(current_rot, target_rot)) ** 0.5
                
                # 角度差に基づく報酬
                if angle_diff < 0.5:  # 差が小さい場合
                    reward += 0.4
                elif angle_diff < 1.0:
                    reward += 0.2
        
        return max(0.0, min(1.0, reward))
    
    def _execute_joint_control_action(self, parameters: Dict[str, Any]) -> float:
        """
        関節制御行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        joint_id = parameters.get("joint_id", 0)
        angle = parameters.get("angle", 0.0)
        force = parameters.get("force", 10.0)
        
        # 関節角度の更新
        if 0 <= joint_id < len(self.state["joint_angles"]):
            self.state["joint_angles"][joint_id] = angle
        else:
            # 関節IDが範囲外の場合は関節角度を拡張
            while len(self.state["joint_angles"]) <= joint_id:
                self.state["joint_angles"].append(0.0)
            self.state["joint_angles"][joint_id] = angle
        
        # 動作精度に基づく報酬
        reward = 0.3
        
        # 目標姿勢が指定されている場合は姿勢に基づいて報酬を調整
        for goal in self.state["goals"]:
            if goal.get("type") == "pose" and "joint_angles" in goal:
                target_angles = goal["joint_angles"]
                
                # 対応する関節角度の確認
                if joint_id < len(target_angles):
                    target_angle = target_angles[joint_id]
                    
                    # 角度差の計算
                    angle_diff = abs(angle - target_angle)
                    
                    # 角度差に基づく報酬
                    if angle_diff < 0.1:  # 差が小さい場合
                        reward += 0.4
                    elif angle_diff < 0.3:
                        reward += 0.2
        
        return max(0.0, min(1.0, reward))
    
    def _execute_speak_action(self, parameters: Dict[str, Any]) -> float:
        """
        発話行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        text = parameters.get("text", "")
        volume = parameters.get("volume", 1.0)
        emotion = parameters.get("emotion", "neutral")
        
        # ログに発話を記録
        logger.info(f"ロボット発話: {text} (感情: {emotion}, 音量: {volume})")
        
        # 会話目標に基づく報酬
        reward = 0.3  # デフォルトの報酬
        
        # 対話目標がある場合は内容に基づいて報酬を調整
        for goal in self.state["goals"]:
            if goal.get("type") == "conversation" and "keywords" in goal:
                keywords = goal["keywords"]
                
                # キーワードの一致数をカウント
                matches = sum(1 for keyword in keywords if keyword.lower() in text.lower())
                
                # 一致数に基づく報酬
                if matches > 0:
                    keyword_reward = min(0.5, matches * 0.1)
                    reward += keyword_reward
        
        return max(0.0, min(1.0, reward))
    
    def _execute_grasp_action(self, parameters: Dict[str, Any]) -> float:
        """
        把持行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        target_id = parameters.get("target_id", None)
        hand = parameters.get("hand", "right")
        force = parameters.get("force", 5.0)
        
        # ログに把持行動を記録
        logger.info(f"把持行動: ターゲット {target_id}, 手: {hand}, 力: {force}")
        
        # 把持対象の検索
        grasp_success = False
        grasp_distance = float('inf')
        
        objects = self.state["sensor_data"].get("objects", [])
        for obj in objects:
            if obj.get("id") == target_id or target_id is None:
                # 距離の計算
                obj_pos = obj.get("position", [0, 0, 0])
                distance = sum((a - b) ** 2 for a, b in zip(self.state["position"], obj_pos)) ** 0.5
                
                # 把持可能距離内であれば把持成功
                if distance < 1.0:  # 1.0メートル以内
                    grasp_success = True
                    grasp_distance = distance
                    break
        
        # 報酬の計算
        if grasp_success:
            # 距離が近いほど高い報酬
            reward = 0.5 + max(0.0, 0.5 * (1.0 - grasp_distance / 1.0))
        else:
            reward = 0.1  # 失敗時の最小報酬
        
        return max(0.0, min(1.0, reward))
    
    def _execute_release_action(self, parameters: Dict[str, Any]) -> float:
        """
        解放行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        hand = parameters.get("hand", "right")
        
        # ログに解放行動を記録
        logger.info(f"解放行動: 手: {hand}")
        
        # 常に成功と見なす単純な実装
        return 0.4
    
    def _execute_composite_action(self, parameters: Dict[str, Any]) -> float:
        """
        複合行動の実行
        
        Args:
            parameters: 行動パラメータ
        
        Returns:
            報酬値
        """
        actions = parameters.get("actions", [])
        if not actions:
            return 0.0
        
        # 各サブアクションを実行し、累積報酬を計算
        total_reward = 0.0
        count = 0
        
        for action in actions:
            action_type = action.get("type", "unknown")
            action_params = action.get("parameters", {})
            
            # 行動タイプに基づいて適切な実行メソッドを呼び出す
            if action_type == "move":
                reward = self._execute_move_action(action_params)
            elif action_type == "rotate":
                reward = self._execute_rotate_action(action_params)
            elif action_type == "joint_control":
                reward = self._execute_joint_control_action(action_params)
            elif action_type == "speak":
                reward = self._execute_speak_action(action_params)
            elif action_type == "grasp":
                reward = self._execute_grasp_action(action_params)
            elif action_type == "release":
                reward = self._execute_release_action(action_params)
            else:
                reward = 0.0
            
            total_reward += reward
            count += 1
        
        # 平均報酬を返す
        return total_reward / max(1, count)

    def change_llm_provider(self, provider: str, **kwargs) -> bool:
        """
        LLMプロバイダーを変更します。
        
        Args:
            provider: 新しいプロバイダー名 ("openai", "claude", "gemini", "vertex", "local")
            **kwargs: プロバイダー固有の設定
            
        Returns:
            変更が成功したかどうか
        """
        if not HAVE_LLM_AGENT or self.agent is None:
            logger.error("LLMエージェントが利用できないため、プロバイダーを変更できません。")
            return False
        
        success = self.agent.change_provider(provider, **kwargs)
        
        if success:
            # 設定の更新
            self.llm_config["provider"] = provider
            for key, value in kwargs.items():
                if key in self.llm_config:
                    self.llm_config[key] = value
                
            logger.info(f"LLMプロバイダーを {provider} に変更しました。")
        else:
            logger.error(f"LLMプロバイダーの変更に失敗しました。")
        
        return success 