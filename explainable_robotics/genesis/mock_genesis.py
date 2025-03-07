"""
Genesisライブラリのモック実装

実際のGenesisライブラリがインストールされていない場合に使用されます。
`import genesis as gs`と互換性のある形式で提供されます。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import sys
import types

class Scene:
    """シーンのモッククラス"""
    
    def __init__(self):
        self.models = []
        self.window_width = 800
        self.window_height = 600
        self.window_title = "Mock Genesis Scene"
        self.camera = None
        self.physics = None
        self.ui = None
    
    def set_window_size(self, width, height):
        """ウィンドウサイズの設定"""
        self.window_width = width
        self.window_height = height
    
    def set_window_title(self, title):
        """ウィンドウタイトルの設定"""
        self.window_title = title
    
    def add_model(self, model):
        """モデルの追加"""
        self.models.append(model)
    
    def set_camera(self, camera):
        """カメラの設定"""
        self.camera = camera
    
    def set_physics(self, physics):
        """物理エンジンの設定"""
        self.physics = physics
    
    def set_ui(self, ui):
        """UIの設定"""
        self.ui = ui
    
    def start(self):
        """シーンの開始"""
        print(f"モックシーンを開始: {self.window_title} ({self.window_width}x{self.window_height})")
    
    def stop(self):
        """シーンの停止"""
        print("モックシーンを停止")

class Camera:
    """カメラのモッククラス"""
    
    def __init__(self):
        self.position = [0, 0, 5]
        self.target = [0, 0, 0]
        self.up = [0, 1, 0]
        self.fov = 45.0
        self.mode = "fixed"
    
    def set_position(self, position):
        """位置の設定"""
        self.position = position
    
    def set_target(self, target):
        """注視点の設定"""
        self.target = target
    
    def set_up(self, up):
        """上方向の設定"""
        self.up = up
    
    def set_fov(self, fov):
        """視野角の設定"""
        self.fov = fov
    
    def set_follow_mode(self, target, distance, up):
        """追従モードの設定"""
        self.mode = "follow"
        self.target = target
        self.distance = distance
        self.up = up
    
    def set_first_person_mode(self, position, forward, up):
        """一人称モードの設定"""
        self.mode = "first_person"
        self.position = position
        self.forward = forward
        self.up = up

class Model:
    """モデルのモッククラス"""
    
    def __init__(self):
        self.position = [0, 0, 0]
        self.rotation = [0, 0, 0]
        self.joints = 20
    
    def set_position(self, position):
        """位置の設定"""
        self.position = position
    
    def set_rotation(self, rotation):
        """回転の設定"""
        self.rotation = rotation
    
    def set_joint_angle(self, index, angle):
        """関節角度の設定"""
        pass
    
    def get_joint_count(self):
        """関節数の取得"""
        return self.joints
    
    @classmethod
    def from_urdf(cls, path):
        """URDFからモデルを作成"""
        print(f"URDFからモデルを作成: {path}")
        return cls()
    
    @classmethod
    def create_humanoid(cls):
        """ヒューマノイドモデルを作成"""
        print("ヒューマノイドモデルを作成")
        return cls()

class UI:
    """UIのモッククラス"""
    
    def __init__(self):
        self.panels = {}
        self.elements = {}
    
    def add_panel(self, title, position, size):
        """パネルの追加"""
        self.panels[title] = {"position": position, "size": size, "elements": []}
    
    def add_gauge(self, panel, title, min_value, max_value, initial_value):
        """ゲージの追加"""
        if panel in self.panels:
            element_id = f"{panel}_{title}"
            self.elements[element_id] = {"type": "gauge", "value": initial_value}
    
    def add_text(self, panel, title, text):
        """テキストの追加"""
        if panel in self.panels:
            element_id = f"{panel}_{title}"
            self.elements[element_id] = {"type": "text", "value": text}
    
    def add_button(self, panel, title, callback):
        """ボタンの追加"""
        if panel in self.panels:
            element_id = f"{panel}_{title}"
            self.elements[element_id] = {"type": "button", "callback": callback}
    
    def add_checkbox(self, panel, title, checked, callback):
        """チェックボックスの追加"""
        if panel in self.panels:
            element_id = f"{panel}_{title}"
            self.elements[element_id] = {"type": "checkbox", "value": checked, "callback": callback}
    
    def update_gauge(self, panel, title, value):
        """ゲージの更新"""
        element_id = f"{panel}_{title}"
        if element_id in self.elements:
            self.elements[element_id]["value"] = value
    
    def update_text(self, panel, title, text):
        """テキストの更新"""
        element_id = f"{panel}_{title}"
        if element_id in self.elements:
            self.elements[element_id]["value"] = text

class Physics:
    """物理エンジンのモッククラス"""
    
    def __init__(self):
        self.gravity = [0, -9.81, 0]
        self.time_step = 0.01
    
    def set_gravity(self, gravity):
        """重力の設定"""
        self.gravity = gravity
    
    def step(self, dt):
        """物理シミュレーションの1ステップ"""
        pass

class Environment:
    """環境のモッククラス"""
    
    def __init__(self, **kwargs):
        self.gravity = kwargs.get("gravity", [0, 0, -9.81])
        self.ground = kwargs.get("ground", True)
        self.sky = kwargs.get("sky", True)
        self.shadows = kwargs.get("shadows", True)
        self.robots = []
    
    def add_robot(self, robot):
        """ロボットを環境に追加"""
        self.robots.append(robot)
    
    def step(self):
        """環境を1ステップ進める"""
        for robot in self.robots:
            if hasattr(robot, 'update'):
                robot.update()
    
    def close(self):
        """環境を閉じる"""
        pass

class HumanoidRobot:
    """ヒューマノイドロボットのモッククラス"""
    
    def __init__(self, **kwargs):
        self.height = kwargs.get("height", 1.7)
        self.mass = kwargs.get("mass", 70.0)
        self.joint_limits = kwargs.get("joint_limits", {})
        
        # ロボットの状態
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.joint_angles = np.zeros(20)
    
    def set_position(self, position):
        """位置の設定"""
        self.position = np.array(position)
    
    def set_orientation(self, orientation):
        """姿勢の設定"""
        self.orientation = np.array(orientation)
    
    def set_joint_angles(self, joint_angles):
        """関節角度の設定"""
        self.joint_angles = np.array(joint_angles)
    
    def update(self):
        """ロボットの状態を更新"""
        # 実際のシミュレーションでは物理エンジンによる更新が行われる
        pass

class Viewer:
    """可視化ビューワーのモッククラス"""
    
    def __init__(self, env):
        self.env = env
        self.camera_position = [3, 3, 2]
        self.camera_target = [0, 0, 1]
        self.camera_up = [0, 0, 1]
    
    def set_camera(self, position, target, up):
        """カメラの設定"""
        self.camera_position = position
        self.camera_target = target
        self.camera_up = up
    
    def update(self):
        """ビューワーの更新"""
        pass
    
    def render(self):
        """シーンのレンダリング"""
        print("シーンをレンダリングしています...")
        for robot in self.env.robots:
            print(f"  ロボット位置: {robot.position}")
            print(f"  ロボット姿勢: {robot.orientation}")
    
    def save_screenshot(self, filename):
        """スクリーンショットの保存"""
        print(f"スクリーンショットを保存: {filename}")
    
    def close(self):
        """ビューワーを閉じる"""
        pass

class NeurotransmitterSystem:
    """神経伝達物質システムのモッククラス"""
    
    def __init__(self):
        self.levels = {
            'dopamine': 0.5,
            'serotonin': 0.5,
            'acetylcholine': 0.5,
            'noradrenaline': 0.5,
            'glutamate': 0.5,
            'gaba': 0.5
        }
    
    def set_level(self, transmitter_type, level, target_regions=None):
        """神経伝達物質レベルの設定"""
        if transmitter_type in self.levels:
            self.levels[transmitter_type] = level
    
    def get_level(self, transmitter_type):
        """神経伝達物質レベルの取得"""
        return self.levels.get(transmitter_type, 0.5)

# ===== モジュール構造を正しく設定 =====

# まず、gsとgensis両方のモジュールを作成
gs = types.ModuleType('genesis')
sys.modules['genesis'] = gs

# サブモジュールの作成
humanoid = types.ModuleType('genesis.humanoid')
sensors = types.ModuleType('genesis.sensors')
motors = types.ModuleType('genesis.motors')
kinematics = types.ModuleType('genesis.kinematics')

# サブモジュールをgsに登録
gs.humanoid = humanoid
gs.sensors = sensors
gs.motors = motors
gs.kinematics = kinematics

# トップレベルクラスをgsに追加
gs.Scene = Scene
gs.Camera = Camera
gs.Model = Model
gs.UI = UI
gs.Physics = Physics
gs.Environment = Environment

# クラスをサブモジュールに登録
humanoid.HumanoidRobot = HumanoidRobot

# センサー類
sensors.Camera = type('Camera', (), {})
sensors.IMU = type('IMU', (), {})
sensors.JointSensor = type('JointSensor', (), {})
sensors.ForceSensor = type('ForceSensor', (), {})

# モーター類
motors.ServoMotor = type('ServoMotor', (), {})

# キネマティクス
kinematics.InverseKinematics = type('InverseKinematics', (), {})

# 後方互換性のためのエクスポート
visualization = type('', (), {'Viewer': Viewer})()
robot = type('', (), {'HumanoidRobot': HumanoidRobot})()
motor = type('', (), {})()
neurotransmitters = type('', (), {'NeurotransmitterSystem': NeurotransmitterSystem})() 