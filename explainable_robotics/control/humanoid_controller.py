"""
ヒューマノイドロボット制御モジュール

このモジュールは説明可能な大脳皮質モデルを用いたヒューマノイドロボット制御クラスを提供します。
Genesis物理シミュレーションライブラリと統合され、ロボットの動作を生成・制御します。
"""

import torch
import numpy as np
import time
import threading
import os
import json
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# 相対インポート
from ..cortical.model import CorticalModel, CorticalBioKAN
from ..utils.logging import get_logger, ActionLogger
from ..utils.explanation import generate_explanation, save_explanation
from ..utils.conversion import tensor_to_numpy, numpy_to_tensor

# Genesis物理エンジンとの統合（利用可能な場合）
try:
    import genesis
    import genesis.motor as gmotor
    import genesis.humanoid as ghumanoid
    from genesis.sensors import SensorArray
    GENESIS_AVAILABLE = True
except ImportError:
    print("警告: Genesisライブラリが見つかりません。シミュレーションモードで動作します。")
    GENESIS_AVAILABLE = False
    # モッククラス
    class GMotor:
        def set_angle(self, joint_id, angle):
            pass
        def set_torque(self, joint_id, torque):
            pass
    
    class GHumanoid:
        def __init__(self):
            self.motors = GMotor()
            self.joint_ids = list(range(20))  # 20関節と仮定
            
        def get_joint_positions(self):
            return np.zeros(len(self.joint_ids))
        
        def get_sensor_data(self):
            return {'accelerometer': np.zeros(3), 'gyroscope': np.zeros(3)}
    
    class SensorArray:
        def __init__(self):
            pass
        def read(self):
            return {'camera': np.zeros((64, 64, 3)), 'depth': np.zeros((64, 64))}

# ロガーの設定
logger = get_logger(__name__)

class HumanoidController:
    """
    説明可能な皮質モデルを使用したヒューマノイドロボットコントローラ
    
    大脳皮質の層構造を模倣したモデルを用いて、ロボットの動作を制御します。
    各行動に対する説明を生成し、ロボットの意思決定プロセスを理解可能にします。
    """
    
    def __init__(
        self,
        model: Union[CorticalModel, CorticalBioKAN],
        simulation_mode: bool = not GENESIS_AVAILABLE,
        robot_config: Optional[Dict[str, Any]] = None,
        max_action_history: int = 100,
        explanation_dir: str = 'explanations'
    ):
        """
        初期化
        
        Args:
            model: 大脳皮質モデル
            simulation_mode: シミュレーションモードで動作するかどうか
            robot_config: ロボット設定
            max_action_history: 保持する行動履歴の最大数
            explanation_dir: 説明保存ディレクトリ
        """
        self.model = model
        self.simulation_mode = simulation_mode
        self.config = robot_config or {}
        self.explanation_dir = explanation_dir
        
        # 行動ロガー
        self.action_logger = ActionLogger('robot_actions', max_action_history)
        
        # 実行状態
        self.running = False
        self.control_thread = None
        self.control_interval = self.config.get('control_interval', 0.1)  # 10Hz
        
        # モデル入力のシェイプ
        self.input_dim = self.config.get('input_dim', 64)
        
        # モータ出力のシェイプ
        self.output_dim = getattr(self.model, 'output_dim', 20)  # デフォルトは20関節
        
        # ヒューマノイドロボットの初期化
        if not self.simulation_mode and GENESIS_AVAILABLE:
            self._init_robot()
        else:
            # シミュレーションモード
            self.robot = GHumanoid() if not GENESIS_AVAILABLE else None
            self.sensors = SensorArray()
            logger.info("シミュレーションモードで初期化しました")
        
        # 説明ディレクトリの作成
        if not os.path.exists(self.explanation_dir):
            os.makedirs(self.explanation_dir)
        
        # 神経伝達物質レベル
        self.neurotransmitter_levels = {
            'acetylcholine': 0.5,  # 注意力と記憶
            'dopamine': 0.5,       # 報酬と動機づけ
            'serotonin': 0.5,      # 気分と情緒
            'noradrenaline': 0.5   # 覚醒と注意
        }
        
        logger.info(f"ヒューマノイドコントローラを初期化しました: 入力次元={self.input_dim}, 出力次元={self.output_dim}")
    
    def _init_robot(self):
        """実際のロボットまたはシミュレーションを初期化"""
        if not GENESIS_AVAILABLE:
            logger.error("Genesisライブラリが利用できないため、ロボットを初期化できません")
            return
        
        try:
            # ロボットモデルの読み込み
            model_path = self.config.get('model_path', 'humanoid_model.json')
            self.robot = ghumanoid.HumanoidRobot(model_path)
            
            # センサーの初期化
            sensor_config = self.config.get('sensors', {
                'camera': {'resolution': (64, 64)},
                'depth': {'resolution': (64, 64)},
                'imu': True
            })
            self.sensors = SensorArray(sensor_config)
            
            logger.info("ロボットを初期化しました")
        except Exception as e:
            logger.error(f"ロボット初期化エラー: {str(e)}")
            self.simulation_mode = True
            self.robot = GHumanoid()
            self.sensors = SensorArray()
    
    def start(self):
        """制御ループを開始"""
        if self.running:
            logger.warning("制御ループは既に実行中です")
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("制御ループを開始しました")
    
    def stop(self):
        """制御ループを停止"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
            self.control_thread = None
        
        logger.info("制御ループを停止しました")
    
    def _control_loop(self):
        """メイン制御ループ"""
        while self.running:
            try:
                # センサーデータを取得
                sensor_data = self._get_sensor_data()
                
                # モデル入力を準備
                model_input = self._prepare_model_input(sensor_data)
                
                # モデルから行動を取得
                motor_commands = self._get_model_output(model_input)
                
                # ロボットに行動を適用
                self._apply_motor_commands(motor_commands)
                
                # 行動を記録
                self._log_action(motor_commands, model_input)
                
                # 制御間隔で待機
                time.sleep(self.control_interval)
                
            except Exception as e:
                logger.error(f"制御ループエラー: {str(e)}")
                time.sleep(1.0)  # エラー時は少し長く待機
    
    def _get_sensor_data(self) -> Dict[str, np.ndarray]:
        """センサーデータを取得"""
        if self.simulation_mode:
            # シミュレーションモードではダミーデータを生成
            return {
                'camera': np.random.rand(64, 64, 3),
                'depth': np.random.rand(64, 64),
                'imu': {
                    'accelerometer': np.random.rand(3) * 2 - 1,  # -1〜1
                    'gyroscope': np.random.rand(3) * 2 - 1       # -1〜1
                },
                'joint_positions': np.random.rand(self.output_dim) * 2 - 1  # -1〜1
            }
        
        # 実際のロボットからデータを取得
        sensor_data = {}
        
        # センサーアレイからデータを読み取り
        array_data = self.sensors.read()
        for key, value in array_data.items():
            sensor_data[key] = value
        
        # IMUデータ
        sensor_data['imu'] = {
            'accelerometer': self.robot.get_sensor_data().get('accelerometer', np.zeros(3)),
            'gyroscope': self.robot.get_sensor_data().get('gyroscope', np.zeros(3))
        }
        
        # 関節位置
        sensor_data['joint_positions'] = self.robot.get_joint_positions()
        
        return sensor_data
    
    def _prepare_model_input(self, sensor_data: Dict[str, Any]) -> torch.Tensor:
        """モデル入力を準備"""
        # 特徴抽出（シンプルな実装）
        features = []
        
        # IMUデータ
        if 'imu' in sensor_data:
            imu_data = sensor_data['imu']
            features.append(imu_data['accelerometer'].flatten())
            features.append(imu_data['gyroscope'].flatten())
        
        # 関節位置
        if 'joint_positions' in sensor_data:
            features.append(sensor_data['joint_positions'].flatten())
        
        # カメラデータ（簡易的な処理）
        if 'camera' in sensor_data:
            # 平均色情報のみ使用
            camera_features = np.mean(sensor_data['camera'], axis=(0, 1))
            features.append(camera_features)
        
        # 深度データ（簡易的な処理）
        if 'depth' in sensor_data:
            # 4分割した領域の平均深度
            h, w = sensor_data['depth'].shape
            depth_features = [
                np.mean(sensor_data['depth'][:h//2, :w//2]),
                np.mean(sensor_data['depth'][:h//2, w//2:]),
                np.mean(sensor_data['depth'][h//2:, :w//2]),
                np.mean(sensor_data['depth'][h//2:, w//2:])
            ]
            features.append(np.array(depth_features))
        
        # 特徴ベクトルを結合
        feature_vector = np.concatenate([f.flatten() for f in features])
        
        # 入力サイズに合わせて調整
        if len(feature_vector) < self.input_dim:
            # 足りない部分はゼロパディング
            padding = np.zeros(self.input_dim - len(feature_vector))
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > self.input_dim:
            # 余分な部分は切り捨て
            feature_vector = feature_vector[:self.input_dim]
        
        # テンソルに変換
        return numpy_to_tensor(feature_vector.reshape(1, -1))
    
    def _get_model_output(self, model_input: torch.Tensor) -> np.ndarray:
        """モデルから行動を取得"""
        with torch.no_grad():
            # 神経伝達物質レベルをコンテキストとして渡す
            context = {k: v for k, v in self.neurotransmitter_levels.items()}
            
            # モデル出力を取得
            model_output = self.model(model_input, context)
            
            # NumPy配列に変換
            output_np = tensor_to_numpy(model_output)
            
            return output_np.reshape(-1)  # バッチ次元を削除
    
    def _apply_motor_commands(self, motor_commands: np.ndarray):
        """モータコマンドをロボットに適用"""
        if self.simulation_mode:
            # シミュレーションモードでは実際の適用はしない
            logger.debug(f"モータコマンド（シミュレーション）: {motor_commands}")
            return
        
        # Genesisロボットに適用
        try:
            for i, command in enumerate(motor_commands):
                if i < len(self.robot.joint_ids):
                    # 値を-1〜1から適切な角度範囲に変換
                    angle = command * 90.0  # -90度〜90度
                    self.robot.motors.set_angle(self.robot.joint_ids[i], angle)
        except Exception as e:
            logger.error(f"モータコマンド適用エラー: {str(e)}")
    
    def _log_action(self, motor_commands: np.ndarray, sensor_input: torch.Tensor):
        """行動をログに記録し、説明を生成"""
        # モデルが説明可能インターフェースを持っている場合
        if hasattr(self.model, 'explain_action'):
            explanation = self.model.explain_action(sensor_input)
            
            # アクションの値を記録
            action_type = "movement"
            action_value = motor_commands.tolist()
            reason = explanation.get('output_interpretation', '')
            
            # ログに記録
            self.action_logger.log_action(
                action_type=action_type,
                action_value=action_value,
                reason=reason
            )
            
            # 定期的に説明を保存（10アクションごと）
            if len(self.action_logger.action_history) % 10 == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.explanation_dir}/explanation_{timestamp}.json"
                save_explanation(explanation, filename)
        else:
            # 説明機能がない場合は単純に記録
            self.action_logger.log_action(
                action_type="movement",
                action_value=motor_commands.tolist(),
                reason="モデルは説明機能を提供していません"
            )
    
    def set_neurotransmitter_level(self, transmitter: str, level: float):
        """
        神経伝達物質レベルを設定
        
        Args:
            transmitter: 神経伝達物質名（'acetylcholine', 'dopamine', 'serotonin', 'noradrenaline'）
            level: レベル値（0〜1）
        """
        if transmitter in self.neurotransmitter_levels:
            # 値を0〜1の範囲に制限
            level = max(0.0, min(1.0, level))
            self.neurotransmitter_levels[transmitter] = level
            
            # モデルに適用（互換性がある場合）
            if hasattr(self.model, 'modulate_neurotransmitter'):
                self.model.modulate_neurotransmitter(transmitter, level)
            
            logger.info(f"神経伝達物質 {transmitter} のレベルを {level} に設定しました")
        else:
            logger.warning(f"未知の神経伝達物質です: {transmitter}")
    
    def get_neurotransmitter_level(self, transmitter: str) -> float:
        """
        神経伝達物質レベルを取得
        
        Args:
            transmitter: 神経伝達物質名
            
        Returns:
            レベル値（0〜1）
        """
        return self.neurotransmitter_levels.get(transmitter, 0.0)
    
    def get_action_history(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        行動履歴を取得
        
        Args:
            n: 取得する履歴の数（Noneの場合はすべて）
            
        Returns:
            行動履歴
        """
        return self.action_logger.get_history(n)
    
    def save_action_history(self, filename: str):
        """
        行動履歴を保存
        
        Args:
            filename: 保存先のファイル名
        """
        self.action_logger.save_history(filename)
        logger.info(f"行動履歴を保存しました: {filename}")
    
    def explain_current_state(self) -> Dict[str, Any]:
        """
        現在の状態を説明
        
        Returns:
            説明情報を含む辞書
        """
        # センサーデータを取得
        sensor_data = self._get_sensor_data()
        
        # モデル入力を準備
        model_input = self._prepare_model_input(sensor_data)
        
        # 説明を生成
        if hasattr(self.model, 'explain_action'):
            explanation = self.model.explain_action(model_input)
        else:
            # 説明機能がない場合はデフォルトの説明
            explanation = {
                'narrative': "モデルは説明機能を提供していません",
                'neurotransmitter_levels': self.neurotransmitter_levels.copy()
            }
        
        # 現在のセンサー情報と神経伝達物質レベルを追加
        explanation['sensor_summary'] = {
            'imu_accel': sensor_data.get('imu', {}).get('accelerometer', [0, 0, 0]),
            'imu_gyro': sensor_data.get('imu', {}).get('gyroscope', [0, 0, 0]),
        }
        explanation['neurotransmitter_levels'] = self.neurotransmitter_levels.copy()
        
        return explanation 