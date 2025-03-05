import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import json

# Genesis APIをインポート（実際の環境に合わせて調整が必要）
try:
    import genesis
    from genesis.humanoid import HumanoidRobot
    from genesis.sensors import Camera, IMU, JointSensor, ForceSensor
    from genesis.motors import ServoMotor
    from genesis.kinematics import InverseKinematics
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    # モックオブジェクトとしてのダミークラス
    class HumanoidRobot:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class GenesisRobotInterface:
    """
    Genesisヒューマノイドロボットライブラリとの統合インターフェース。
    皮質モデルからの出力をロボットの物理的動作に変換し、
    センサーデータを皮質モデルへの入力に変換します。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # ロボット設定
        self.config = {
            'robot_model': 'humanoid_standard',
            'connection': {
                'type': 'usb',  # usb, wifi, bluetooth
                'port': '/dev/ttyUSB0',
                'baudrate': 115200
            },
            'sensors': {
                'camera': True,
                'imu': True,
                'joint_sensors': True,
                'force_sensors': True
            },
            'motors': {
                'head': ['neck_pitch', 'neck_yaw'],
                'arms': ['shoulder_pitch_l', 'shoulder_roll_l', 'elbow_l', 'wrist_l',
                         'shoulder_pitch_r', 'shoulder_roll_r', 'elbow_r', 'wrist_r'],
                'legs': ['hip_yaw_l', 'hip_roll_l', 'hip_pitch_l', 'knee_l', 'ankle_pitch_l', 'ankle_roll_l',
                         'hip_yaw_r', 'hip_roll_r', 'hip_pitch_r', 'knee_r', 'ankle_pitch_r', 'ankle_roll_r']
            },
            'control': {
                'rate': 50,  # Hz
                'safety_limits': True,
                'motion_smoothing': True
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
                logger.error(f"ロボット設定ファイルの読み込みに失敗しました: {str(e)}")
                
        # Genesisライブラリが利用可能かチェック
        self.robot = None
        self.sensors = {}
        self.motors = {}
        
        if GENESIS_AVAILABLE:
            self._initialize_robot()
        else:
            logger.warning("Genesisライブラリが見つかりません。シミュレーションモードで動作します。")
            self._initialize_simulation()
            
        # 最後のセンサー値とモーター出力を保存
        self.last_sensor_data = {}
        self.last_motor_commands = {}
        
        # コントロールループ用の変数
        self.is_running = False
        self.control_callback = None
        
    def _merge_config(self, new_config: Dict) -> None:
        """設定を再帰的にマージします"""
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self._merge_config(value)
                else:
                    self.config[key] = value
                    
    def _initialize_robot(self) -> None:
        """実際のロボットハードウェアとの接続を初期化します"""
        try:
            # ロボットの初期化
            model = self.config['robot_model']
            conn_type = self.config['connection']['type']
            
            if conn_type == 'usb':
                port = self.config['connection']['port']
                baudrate = self.config['connection']['baudrate']
                self.robot = HumanoidRobot(model=model, port=port, baudrate=baudrate)
            elif conn_type == 'wifi':
                # WiFi接続パラメータ
                host = self.config['connection'].get('host', '192.168.1.1')
                port = self.config['connection'].get('port', 8080)
                self.robot = HumanoidRobot(model=model, host=host, port=port, connection_type='wifi')
            else:
                # その他の接続方法
                self.robot = HumanoidRobot(model=model)
                
            logger.info(f"ロボット {model} を初期化しました（接続: {conn_type}）")
            
            # センサーの初期化
            if self.config['sensors']['camera']:
                self.sensors['camera'] = self.robot.get_camera()
            if self.config['sensors']['imu']:
                self.sensors['imu'] = self.robot.get_imu()
            if self.config['sensors']['joint_sensors']:
                joint_sensors = {}
                # 頭部
                for motor_name in self.config['motors']['head']:
                    joint_sensors[motor_name] = self.robot.get_joint_sensor(motor_name)
                # 腕
                for motor_name in self.config['motors']['arms']:
                    joint_sensors[motor_name] = self.robot.get_joint_sensor(motor_name)
                # 脚
                for motor_name in self.config['motors']['legs']:
                    joint_sensors[motor_name] = self.robot.get_joint_sensor(motor_name)
                self.sensors['joints'] = joint_sensors
                
            if self.config['sensors']['force_sensors']:
                self.sensors['force_left'] = self.robot.get_force_sensor('left_foot')
                self.sensors['force_right'] = self.robot.get_force_sensor('right_foot')
                
            # モーターの初期化
            motors = {}
            # 頭部
            for motor_name in self.config['motors']['head']:
                motors[motor_name] = self.robot.get_motor(motor_name)
            # 腕
            for motor_name in self.config['motors']['arms']:
                motors[motor_name] = self.robot.get_motor(motor_name)
            # 脚
            for motor_name in self.config['motors']['legs']:
                motors[motor_name] = self.robot.get_motor(motor_name)
            self.motors = motors
            
            # 安全制限の設定
            if self.config['control']['safety_limits']:
                self.robot.enable_safety_limits()
                
            logger.info("ロボットのセンサーとモーターを初期化しました")
                
        except Exception as e:
            logger.error(f"ロボットの初期化に失敗しました: {str(e)}")
            # シミュレーションモードにフォールバック
            self._initialize_simulation()
            
    def _initialize_simulation(self) -> None:
        """シミュレーションモードの初期化"""
        # モックデータ構造を作成
        self.robot = "SIMULATION"
        
        # センサーのシミュレーション
        self.sensors = {
            'camera': {'resolution': (640, 480), 'fps': 30},
            'imu': {'gyro': np.zeros(3), 'accel': np.array([0, 0, 9.81])},
            'joints': {}
        }
        
        # 関節センサーのシミュレーション
        for motor_section in ['head', 'arms', 'legs']:
            for motor_name in self.config['motors'][motor_section]:
                self.sensors['joints'][motor_name] = {'position': 0.0, 'velocity': 0.0, 'load': 0.0}
                
        # 足底センサーのシミュレーション
        self.sensors['force_left'] = {'force': np.zeros(3), 'center_of_pressure': np.zeros(2)}
        self.sensors['force_right'] = {'force': np.zeros(3), 'center_of_pressure': np.zeros(2)}
        
        # モーターのシミュレーション
        self.motors = {}
        for motor_section in ['head', 'arms', 'legs']:
            for motor_name in self.config['motors'][motor_section]:
                self.motors[motor_name] = {'target': 0.0, 'current': 0.0, 'max_speed': 1.0}
                
        logger.info("ロボットシミュレーションモードを初期化しました")
        
    def read_sensors(self) -> Dict[str, Any]:
        """
        全てのセンサーデータを読み取り、統一されたフォーマットで返します。
        
        Returns:
            センサー値を含む辞書
        """
        sensor_data = {}
        
        try:
            if GENESIS_AVAILABLE and self.robot != "SIMULATION":
                # 実際のロボットからセンサーデータを読み取り
                
                # カメラ
                if 'camera' in self.sensors:
                    frame = self.sensors['camera'].read_frame()
                    sensor_data['camera'] = frame
                    
                # IMU
                if 'imu' in self.sensors:
                    gyro = self.sensors['imu'].read_gyro()
                    accel = self.sensors['imu'].read_accel()
                    sensor_data['imu'] = {
                        'gyro': np.array(gyro),
                        'accel': np.array(accel)
                    }
                    
                # 関節センサー
                if 'joints' in self.sensors:
                    joint_data = {}
                    for name, sensor in self.sensors['joints'].items():
                        position = sensor.read_position()
                        velocity = sensor.read_velocity()
                        load = sensor.read_load()
                        joint_data[name] = {
                            'position': position,
                            'velocity': velocity,
                            'load': load
                        }
                    sensor_data['joints'] = joint_data
                    
                # 力センサー
                if 'force_left' in self.sensors and 'force_right' in self.sensors:
                    left_force = self.sensors['force_left'].read_force()
                    left_cop = self.sensors['force_left'].read_center_of_pressure()
                    right_force = self.sensors['force_right'].read_force()
                    right_cop = self.sensors['force_right'].read_center_of_pressure()
                    
                    sensor_data['force'] = {
                        'left': {
                            'force': np.array(left_force),
                            'center_of_pressure': np.array(left_cop)
                        },
                        'right': {
                            'force': np.array(right_force),
                            'center_of_pressure': np.array(right_cop)
                        }
                    }
            else:
                # シミュレーションモードではモックデータを使用
                
                # カメラ（ダミーフレーム生成）
                if 'camera' in self.sensors:
                    resolution = self.sensors['camera']['resolution']
                    sensor_data['camera'] = np.zeros((*resolution, 3), dtype=np.uint8)
                    
                # IMU
                if 'imu' in self.sensors:
                    sensor_data['imu'] = {
                        'gyro': self.sensors['imu']['gyro'].copy(),
                        'accel': self.sensors['imu']['accel'].copy()
                    }
                    
                # 関節データ
                if 'joints' in self.sensors:
                    sensor_data['joints'] = {}
                    for name, data in self.sensors['joints'].items():
                        sensor_data['joints'][name] = data.copy()
                        
                # 力センサー
                sensor_data['force'] = {
                    'left': {
                        'force': self.sensors['force_left']['force'].copy(),
                        'center_of_pressure': self.sensors['force_left']['center_of_pressure'].copy()
                    },
                    'right': {
                        'force': self.sensors['force_right']['force'].copy(),
                        'center_of_pressure': self.sensors['force_right']['center_of_pressure'].copy()
                    }
                }
                
                # シミュレーションに若干のノイズを追加（よりリアルに）
                self._add_simulation_noise(sensor_data)
                
            # 最後のセンサーデータを保存
            self.last_sensor_data = sensor_data
            return sensor_data
            
        except Exception as e:
            logger.error(f"センサーデータの読み取りに失敗しました: {str(e)}")
            # 前回のデータがあればそれを返す、なければ空の辞書
            return self.last_sensor_data if self.last_sensor_data else {}
            
    def _add_simulation_noise(self, sensor_data: Dict[str, Any]) -> None:
        """シミュレーションデータにリアリスティックなノイズを追加します"""
        # IMUノイズ
        if 'imu' in sensor_data:
            sensor_data['imu']['gyro'] += np.random.normal(0, 0.01, 3)
            sensor_data['imu']['accel'] += np.random.normal(0, 0.05, 3)
            
        # 関節センサーノイズ
        if 'joints' in sensor_data:
            for name in sensor_data['joints']:
                sensor_data['joints'][name]['position'] += np.random.normal(0, 0.001)
                sensor_data['joints'][name]['velocity'] += np.random.normal(0, 0.005)
                sensor_data['joints'][name]['load'] += np.random.normal(0, 0.01)
                
        # 力センサーノイズ
        if 'force' in sensor_data:
            sensor_data['force']['left']['force'] += np.random.normal(0, 0.1, 3)
            sensor_data['force']['left']['center_of_pressure'] += np.random.normal(0, 0.002, 2)
            sensor_data['force']['right']['force'] += np.random.normal(0, 0.1, 3)
            sensor_data['force']['right']['center_of_pressure'] += np.random.normal(0, 0.002, 2)
            
    def send_motor_commands(self, motor_commands: Dict[str, float]) -> bool:
        """
        モーターコマンドを送信します。
        
        Args:
            motor_commands: モーター名から目標位置（-1.0〜1.0）へのマッピング
            
        Returns:
            コマンド送信が成功したかどうか
        """
        try:
            # コマンドの範囲チェック
            for motor_name, value in motor_commands.items():
                if value < -1.0 or value > 1.0:
                    logger.warning(f"モーター {motor_name} の値 {value} が範囲外です。-1.0〜1.0に制限します。")
                    motor_commands[motor_name] = max(-1.0, min(1.0, value))
                    
            # モーションスムージングの適用
            if self.config['control']['motion_smoothing'] and self.last_motor_commands:
                smoothed_commands = {}
                smoothing_factor = 0.2  # 小さいほど滑らか（0.0〜1.0）
                
                for motor_name, target in motor_commands.items():
                    if motor_name in self.last_motor_commands:
                        # 前回の値と現在の目標値の間を補間
                        last_value = self.last_motor_commands[motor_name]
                        smoothed_value = last_value + smoothing_factor * (target - last_value)
                        smoothed_commands[motor_name] = smoothed_value
                    else:
                        smoothed_commands[motor_name] = target
                        
                motor_commands = smoothed_commands
                
            # 実際にコマンドを送信
            if GENESIS_AVAILABLE and self.robot != "SIMULATION":
                for motor_name, target in motor_commands.items():
                    if motor_name in self.motors:
                        self.motors[motor_name].set_position(target)
            else:
                # シミュレーションモードでは位置を更新するだけ
                for motor_name, target in motor_commands.items():
                    if motor_name in self.motors:
                        # 目標値に向かって徐々に動かす（シミュレーション）
                        current = self.motors[motor_name]['current']
                        max_speed = self.motors[motor_name]['max_speed']
                        step = min(max_speed, abs(target - current))
                        step *= 1 if target > current else -1
                        self.motors[motor_name]['current'] = current + step
                        self.motors[motor_name]['target'] = target
                        
            # 最後のコマンドを保存
            self.last_motor_commands = motor_commands.copy()
            return True
            
        except Exception as e:
            logger.error(f"モーターコマンドの送信に失敗しました: {str(e)}")
            return False
            
    def convert_cortical_output_to_motor_commands(
        self,
        cortical_output: np.ndarray,
        action_type: str = 'direct'
    ) -> Dict[str, float]:
        """
        皮質モデルの出力をモーターコマンドに変換します。
        
        Args:
            cortical_output: 皮質モデルから出力された値のベクトル
            action_type: 変換タイプ（'direct', 'pose', 'velocity'）
            
        Returns:
            モーターコマンドの辞書
        """
        # 出力の次元を確認
        output_dim = len(cortical_output)
        motor_commands = {}
        
        # 直接制御モード（各出力が直接モーター位置に対応）
        if action_type == 'direct':
            all_motors = []
            for section in ['head', 'arms', 'legs']:
                all_motors.extend(self.config['motors'][section])
                
            # 出力次元とモーター数の整合性をチェック
            if output_dim == len(all_motors):
                for i, motor_name in enumerate(all_motors):
                    motor_commands[motor_name] = float(cortical_output[i])
            else:
                logger.warning(f"出力次元 ({output_dim}) がモーター数 ({len(all_motors)}) と一致しません")
                # フォールバック: 利用可能なモーターにマッピング
                for i, motor_name in enumerate(all_motors[:output_dim]):
                    motor_commands[motor_name] = float(cortical_output[i])
                    
        # ポーズモード（事前定義されたポーズの補間）
        elif action_type == 'pose':
            # 実装例: 定義済みポーズデータベースから選択または補間
            # ここでは単純化のため、直接制御にフォールバック
            return self.convert_cortical_output_to_motor_commands(cortical_output, 'direct')
            
        # 速度モード（関節速度の制御）
        elif action_type == 'velocity':
            # 速度モードの実装
            # この例ではモータの現在位置に速度（cortical_output）を加えた目標位置を計算
            all_motors = []
            for section in ['head', 'arms', 'legs']:
                all_motors.extend(self.config['motors'][section])
                
            if output_dim == len(all_motors):
                for i, motor_name in enumerate(all_motors):
                    if GENESIS_AVAILABLE and self.robot != "SIMULATION":
                        # 実際のロボットから現在位置を取得
                        current_pos = self.motors[motor_name].get_position()
                    else:
                        # シミュレーションの場合
                        current_pos = self.motors[motor_name]['current']
                        
                    # 速度（-1〜1の範囲）を位置の変化量に変換
                    # 速度に0.05を掛けることで、小さな増分にする
                    delta = float(cortical_output[i]) * 0.05
                    target_pos = current_pos + delta
                    
                    # 範囲制限
                    target_pos = max(-1.0, min(1.0, target_pos))
                    motor_commands[motor_name] = target_pos
            else:
                logger.warning(f"出力次元 ({output_dim}) がモーター数 ({len(all_motors)}) と一致しません")
                return {}
        else:
            logger.error(f"未知のアクションタイプ: {action_type}")
            return {}
            
        return motor_commands
        
    def start_control_loop(self, control_callback: Callable[[Dict[str, Any]], np.ndarray]) -> None:
        """
        制御ループを開始します。
        
        Args:
            control_callback: センサーデータを引数に取り、モーターコマンドを返す関数
        """
        if self.is_running:
            logger.warning("制御ループはすでに実行中です")
            return
            
        self.is_running = True
        self.control_callback = control_callback
        
        try:
            logger.info("制御ループを開始します")
            control_rate = self.config['control']['rate']
            period = 1.0 / control_rate
            
            while self.is_running:
                loop_start = time.time()
                
                # センサーデータを読み取り
                sensor_data = self.read_sensors()
                
                # コールバック関数でモーターコマンドを生成
                try:
                    cortical_output = control_callback(sensor_data)
                    motor_commands = self.convert_cortical_output_to_motor_commands(cortical_output)
                    self.send_motor_commands(motor_commands)
                except Exception as e:
                    logger.error(f"コントロールコールバックでエラーが発生しました: {str(e)}")
                    
                # レート制御
                elapsed = time.time() - loop_start
                sleep_time = max(0, period - elapsed)
                if elapsed > period:
                    logger.warning(f"制御ループが目標レート（{control_rate}Hz）に追いついていません")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("ユーザーにより制御ループが停止されました")
        except Exception as e:
            logger.error(f"制御ループでエラーが発生しました: {str(e)}")
        finally:
            self.is_running = False
            logger.info("制御ループを終了しました")
            
    def stop_control_loop(self) -> None:
        """制御ループを停止します"""
        self.is_running = False
        logger.info("制御ループを停止します")
        
    def execute_gesture(self, gesture_name: str) -> bool:
        """
        事前定義されたジェスチャーを実行します。
        
        Args:
            gesture_name: 実行するジェスチャーの名前
            
        Returns:
            ジェスチャーの実行が成功したかどうか
        """
        # 基本的なジェスチャーのマッピング
        gestures = {
            'wave': self._gesture_wave,
            'bow': self._gesture_bow,
            'nod': self._gesture_nod,
            'shake_head': self._gesture_shake_head,
            'point': self._gesture_point,
            'hands_up': self._gesture_hands_up
        }
        
        if gesture_name in gestures:
            try:
                logger.info(f"ジェスチャー '{gesture_name}' を実行します")
                gestures[gesture_name]()
                return True
            except Exception as e:
                logger.error(f"ジェスチャー '{gesture_name}' の実行中にエラーが発生しました: {str(e)}")
                return False
        else:
            logger.warning(f"未知のジェスチャー: {gesture_name}")
            return False
            
    def _gesture_wave(self) -> None:
        """手を振るジェスチャー"""
        # 実装例：右腕を使って手を振る動作
        keyframes = [
            {'shoulder_pitch_r': -0.3, 'shoulder_roll_r': 0.8, 'elbow_r': 0.5, 'wrist_r': 0.0},
            {'shoulder_pitch_r': -0.3, 'shoulder_roll_r': 0.8, 'elbow_r': 0.5, 'wrist_r': 0.5},
            {'shoulder_pitch_r': -0.3, 'shoulder_roll_r': 0.8, 'elbow_r': 0.5, 'wrist_r': -0.5},
            {'shoulder_pitch_r': -0.3, 'shoulder_roll_r': 0.8, 'elbow_r': 0.5, 'wrist_r': 0.5},
            {'shoulder_pitch_r': -0.3, 'shoulder_roll_r': 0.8, 'elbow_r': 0.5, 'wrist_r': -0.5}
        ]
        
        self._execute_keyframes(keyframes, duration=3.0)
        
    def _gesture_bow(self) -> None:
        """お辞儀をするジェスチャー"""
        keyframes = [
            {'hip_pitch_l': 0.0, 'hip_pitch_r': 0.0, 'neck_pitch': 0.0},
            {'hip_pitch_l': 0.4, 'hip_pitch_r': 0.4, 'neck_pitch': 0.3},
            {'hip_pitch_l': 0.0, 'hip_pitch_r': 0.0, 'neck_pitch': 0.0}
        ]
        
        self._execute_keyframes(keyframes, duration=2.0)
        
    def _gesture_nod(self) -> None:
        """頷くジェスチャー"""
        keyframes = [
            {'neck_pitch': 0.0},
            {'neck_pitch': 0.3},
            {'neck_pitch': 0.0},
            {'neck_pitch': 0.3},
            {'neck_pitch': 0.0}
        ]
        
        self._execute_keyframes(keyframes, duration=1.5)
        
    def _gesture_shake_head(self) -> None:
        """首を横に振るジェスチャー"""
        keyframes = [
            {'neck_yaw': 0.0},
            {'neck_yaw': 0.3},
            {'neck_yaw': -0.3},
            {'neck_yaw': 0.3},
            {'neck_yaw': -0.3},
            {'neck_yaw': 0.0}
        ]
        
        self._execute_keyframes(keyframes, duration=2.0)
        
    def _gesture_point(self) -> None:
        """指差しジェスチャー"""
        keyframes = [
            {'shoulder_pitch_r': 0.0, 'shoulder_roll_r': 0.0, 'elbow_r': 0.0},
            {'shoulder_pitch_r': -0.7, 'shoulder_roll_r': 0.2, 'elbow_r': 0.1}
        ]
        
        self._execute_keyframes(keyframes, duration=1.0)
        
    def _gesture_hands_up(self) -> None:
        """両手を上げるジェスチャー"""
        keyframes = [
            {'shoulder_pitch_l': 0.0, 'shoulder_roll_l': 0.0, 'shoulder_pitch_r': 0.0, 'shoulder_roll_r': 0.0},
            {'shoulder_pitch_l': -0.8, 'shoulder_roll_l': 0.3, 'shoulder_pitch_r': -0.8, 'shoulder_roll_r': -0.3}
        ]
        
        self._execute_keyframes(keyframes, duration=1.5)
        
    def _execute_keyframes(self, keyframes: List[Dict[str, float]], duration: float) -> None:
        """
        一連のキーフレームを実行します。
        
        Args:
            keyframes: モーター位置のキーフレームのリスト
            duration: 全キーフレームの実行にかかる合計時間（秒）
        """
        frames_count = len(keyframes)
        if frames_count < 1:
            return
            
        # 各キーフレーム間の時間
        frame_duration = duration / frames_count
        
        for frame in keyframes:
            # キーフレームのモーターコマンドを送信
            self.send_motor_commands(frame)
            # 次のキーフレームまで待機
            time.sleep(frame_duration)
            
    def calibrate(self) -> bool:
        """
        ロボットのセンサーとモーターを校正します。
        
        Returns:
            校正が成功したかどうか
        """
        if GENESIS_AVAILABLE and self.robot != "SIMULATION":
            try:
                logger.info("ロボットの校正を開始します")
                
                # IMUの校正
                if 'imu' in self.sensors:
                    logger.info("IMUを校正しています...")
                    self.sensors['imu'].calibrate()
                    
                # モーターのゼロ位置校正
                logger.info("モーターを校正しています...")
                initial_pose = {}
                for motor_name in self.motors:
                    initial_pose[motor_name] = 0.0
                    
                # 初期姿勢にゆっくり移動
                self.send_motor_commands(initial_pose)
                time.sleep(2.0)  # モーターが移動するのを待つ
                
                # 校正完了
                logger.info("ロボットの校正が完了しました")
                return True
                
            except Exception as e:
                logger.error(f"ロボットの校正中にエラーが発生しました: {str(e)}")
                return False
        else:
            # シミュレーションモードでは単にモーター位置をリセット
            logger.info("シミュレーションモードでロボットを校正しています")
            for motor_name in self.motors:
                if isinstance(self.motors[motor_name], dict):
                    self.motors[motor_name]['current'] = 0.0
                    self.motors[motor_name]['target'] = 0.0
            return True
            
    def shutdown(self) -> None:
        """ロボットを安全にシャットダウンします"""
        try:
            # 制御ループが実行中なら停止
            if self.is_running:
                self.stop_control_loop()
                
            # モーターを安全な位置に
            safe_pose = {}
            for motor_name in self.motors:
                safe_pose[motor_name] = 0.0
                
            logger.info("ロボットを安全姿勢に移動しています...")
            self.send_motor_commands(safe_pose)
            time.sleep(1.0)  # モーターが移動するのを待つ
            
            # Genesisロボットの場合、正式なシャットダウン処理
            if GENESIS_AVAILABLE and self.robot != "SIMULATION":
                logger.info("ロボットをシャットダウンしています...")
                self.robot.shutdown()
                
            logger.info("ロボットのシャットダウンが完了しました")
            
        except Exception as e:
            logger.error(f"ロボットのシャットダウン中にエラーが発生しました: {str(e)}")


def create_robot_interface(config_path: Optional[str] = None) -> GenesisRobotInterface:
    """
    ロボットインターフェースのインスタンスを作成するファクトリ関数
    
    Args:
        config_path: 設定ファイルへのパス（オプション）
        
    Returns:
        初期化されたGenesisRobotInterfaceインスタンス
    """
    return GenesisRobotInterface(config_path) 