"""
ヒューマノイドロボットのコントローラー

ロボットの制御と状態管理を行います。
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from ..utils.logging import get_logger

logger = get_logger(__name__)

class HumanoidController:
    """ヒューマノイドロボットのコントローラー"""
    
    def __init__(
        self,
        num_joints: int = 20,
        max_velocity: float = 1.0,
        max_acceleration: float = 0.5
    ):
        """
        初期化
        
        Args:
            num_joints: 関節の数
            max_velocity: 最大速度
            max_acceleration: 最大加速度
        """
        self.num_joints = num_joints
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        
        # ロボットの状態
        self.state = {
            "position": np.zeros(3),  # x, y, z
            "orientation": np.zeros(3),  # roll, pitch, yaw
            "joint_angles": np.zeros(num_joints),
            "joint_velocities": np.zeros(num_joints),
            "joint_accelerations": np.zeros(num_joints)
        }
        
        # 制御パラメータ
        self.control_params = {
            "kp": 100.0,  # 比例ゲイン
            "kd": 10.0,   # 微分ゲイン
            "ki": 1.0     # 積分ゲイン
        }
        
        # 積分項の初期化
        self.integral = np.zeros(num_joints)
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """
        現在の状態を取得
        
        Returns:
            ロボットの状態
        """
        return self.state.copy()
    
    def execute_action(self, action: Dict[str, float]):
        """
        行動の実行
        
        Args:
            action: 実行する行動（関節角度の目標値）
        """
        try:
            # 行動の検証
            if not self._validate_action(action):
                logger.warning("無効な行動です")
                return
            
            # 目標位置の設定
            target_position = np.array([
                action.get("x", 0.0),
                action.get("y", 0.0),
                action.get("z", 0.0)
            ])
            
            # 目標姿勢の設定
            target_orientation = np.array([
                action.get("roll", 0.0),
                action.get("pitch", 0.0),
                action.get("yaw", 0.0)
            ])
            
            # 目標関節角度の設定
            target_joint_angles = np.array([
                action.get(f"joint_{i}", 0.0)
                for i in range(self.num_joints)
            ])
            
            # 逆運動学の計算
            joint_angles = self._inverse_kinematics(
                target_position,
                target_orientation
            )
            
            # 関節制御
            self._control_joints(joint_angles)
            
            # 状態の更新
            self._update_state()
            
        except Exception as e:
            logger.error(f"行動の実行に失敗: {e}")
    
    def _validate_action(self, action: Dict[str, float]) -> bool:
        """
        行動の検証
        
        Args:
            action: 検証する行動
            
        Returns:
            行動が有効かどうか
        """
        # 位置の制限チェック
        for axis in ["x", "y", "z"]:
            if axis in action:
                if abs(action[axis]) > 1.0:  # 1メートルを超える移動は禁止
                    return False
        
        # 関節角度の制限チェック
        for i in range(self.num_joints):
            joint_key = f"joint_{i}"
            if joint_key in action:
                if abs(action[joint_key]) > np.pi:  # 180度を超える回転は禁止
                    return False
        
        return True
    
    def _inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray
    ) -> np.ndarray:
        """
        逆運動学の計算
        
        Args:
            target_position: 目標位置
            target_orientation: 目標姿勢
            
        Returns:
            関節角度
        """
        # ここでは簡単な実装として、目標位置と姿勢から
        # 関節角度を直接計算する
        # 実際の実装では、より高度な逆運動学アルゴリズムが必要
        
        # 現在の位置と目標位置の差分
        position_diff = target_position - self.state["position"]
        
        # 現在の姿勢と目標姿勢の差分
        orientation_diff = target_orientation - self.state["orientation"]
        
        # 関節角度の計算（簡単な実装）
        joint_angles = np.zeros(self.num_joints)
        
        # 位置と姿勢の差分を関節角度に変換
        for i in range(self.num_joints):
            if i < 3:  # 位置制御用の関節
                joint_angles[i] = position_diff[i % 3]
            elif i < 6:  # 姿勢制御用の関節
                joint_angles[i] = orientation_diff[i - 3]
            else:  # その他の関節は現在の角度を維持
                joint_angles[i] = self.state["joint_angles"][i]
        
        return joint_angles
    
    def _control_joints(self, target_angles: np.ndarray):
        """
        関節の制御
        
        Args:
            target_angles: 目標関節角度
        """
        # 現在の関節角度と目標角度の差分
        error = target_angles - self.state["joint_angles"]
        
        # 積分項の更新
        self.integral += error
        
        # 微分項の計算
        derivative = -self.state["joint_velocities"]
        
        # PID制御
        control = (
            self.control_params["kp"] * error +
            self.control_params["ki"] * self.integral +
            self.control_params["kd"] * derivative
        )
        
        # 加速度の制限
        control = np.clip(
            control,
            -self.max_acceleration,
            self.max_acceleration
        )
        
        # 速度の更新
        self.state["joint_velocities"] += control
        
        # 速度の制限
        self.state["joint_velocities"] = np.clip(
            self.state["joint_velocities"],
            -self.max_velocity,
            self.max_velocity
        )
        
        # 加速度の更新
        self.state["joint_accelerations"] = control
    
    def _update_state(self):
        """状態の更新"""
        # 関節角度の更新
        self.state["joint_angles"] += self.state["joint_velocities"]
        
        # 位置と姿勢の更新（順運動学）
        self._update_position_and_orientation()
    
    def _update_position_and_orientation(self):
        """位置と姿勢の更新"""
        # ここでは簡単な実装として、関節角度から
        # 位置と姿勢を直接計算する
        # 実際の実装では、より高度な順運動学アルゴリズムが必要
        
        # 位置の更新
        self.state["position"] += np.array([
            np.sin(self.state["joint_angles"][0]),
            np.sin(self.state["joint_angles"][1]),
            np.sin(self.state["joint_angles"][2])
        ])
        
        # 姿勢の更新
        self.state["orientation"] += np.array([
            self.state["joint_angles"][3],
            self.state["joint_angles"][4],
            self.state["joint_angles"][5]
        ]) 