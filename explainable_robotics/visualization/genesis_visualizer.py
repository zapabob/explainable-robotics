"""Genesis Visualizer

ヒューマノイドロボットと脳状態の3D可視化を行うモジュール。
物理エンジンを統合し、リアルタイムでの可視化と制御インターフェースを提供します。
"""

import time
import threading
import queue
import os
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np

# 物理エンジンとしてgenesisを使用
GENESIS_AVAILABLE = False
try:
    import genesis as gs
    # クラスの存在を確認
    if (hasattr(gs, 'humanoid') and hasattr(gs, 'Scene') and
        hasattr(gs, 'Camera') and hasattr(gs, 'Model') and
        hasattr(gs, 'UI') and hasattr(gs, 'Physics')):
        GENESIS_AVAILABLE = True
    else:
        print("WARNING: Genesisライブラリ構造が不完全です。モックモードで実行します。")
except ImportError:
    print("WARNING: Genesisライブラリが利用できません。モックモードで実行します。")

# モックモードでのダミークラス
if not GENESIS_AVAILABLE:
    class DummyScene:
        """シーンのダミークラス"""
        def __init__(self):
            pass
        def set_window_size(self, *args):
            pass
        def set_window_title(self, *args):
            pass
        def add_model(self, *args):
            pass
        def set_camera(self, *args):
            pass
        def set_physics(self, *args):
            pass
        def set_ui(self, *args):
            pass
        def start(self):
            print("モックシーンを開始")
        def stop(self):
            print("モックシーンを停止")
    
    # ダミーモジュール
    class DummyModule:
        """すべての属性アクセスに対してダミーオブジェクトを返すモジュール"""
        def __getattr__(self, name):
            return DummyScene()
    
    gs = DummyModule()

class GenesisVisualizer:
    """Genesisを使用したヒューマノイドロボットと脳状態の可視化"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        window_title: str = "Explainable Robotics Visualizer",
        width: int = 1280,
        height: int = 720,
        show_ui: bool = True,
        physics_enabled: bool = True,
        camera_view: str = "follow"
    ):
        """
        初期化
        
        Args:
            model_path: ロボットモデルのパス（URDF形式）
            window_title: ウィンドウタイトル
            width: ウィンドウ幅
            height: ウィンドウ高さ
            show_ui: UIを表示するかどうか
            physics_enabled: 物理シミュレーションを有効にするかどうか
            camera_view: カメラビューモード（'follow', 'fixed', 'first_person'）
        """
        self.model_path = model_path
        self.window_title = window_title
        self.width = width
        self.height = height
        self.show_ui = show_ui
        self.physics_enabled = physics_enabled
        self.camera_view = camera_view
        
        # 内部状態
        self.is_running = False
        self.scene = None
        self.model = None
        self.ui = None
        self.physics = None
        
        # 更新キュー
        self.update_queue = queue.Queue()
        
        # 更新スレッド
        self.update_thread = None
        
        # カメラ設定
        self.camera_settings = {
            'follow': {
                'position': [0, 2, 5],
                'target': [0, 1, 0],
                'up': [0, 1, 0],
                'fov': 45.0,
                'distance': 5.0
            },
            'fixed': {
                'position': [5, 5, 5],
                'target': [0, 0, 0],
                'up': [0, 1, 0],
                'fov': 60.0
            },
            'first_person': {
                'position': [0, 1.7, 0],
                'forward': [0, 0, 1],
                'up': [0, 1, 0],
                'fov': 90.0
            }
        }
        
        # アニメーション設定
        self.animation_settings = {
            'frame_rate': 60,
            'motion_blur': 0.1,
            'interpolation': 'linear'
        }
    
    def start(self):
        """可視化の開始"""
        if self.is_running:
            print("既に実行中です")
            return
        
        # Genesisの利用可否を確認
        if not GENESIS_AVAILABLE:
            print("モックモードで開始します（可視化なし）")
            self.is_running = True
            return
        
        # シーンの初期化
        self.scene = gs.Scene()
        self.scene.set_window_size(self.width, self.height)
        self.scene.set_window_title(self.window_title)
        
        # カメラの設定
        self._setup_camera()
        
        # モデルの読み込み
        if self.model_path and os.path.exists(self.model_path):
            self.model = gs.Model.from_urdf(self.model_path)
            self.scene.add_model(self.model)
        else:
            # デフォルトのヒューマノイドモデル
            self.model = gs.Model.create_humanoid()
            self.scene.add_model(self.model)
        
        # 物理エンジンの初期化
        if self.physics_enabled:
            self.physics = gs.Physics()
            self.physics.set_gravity([0, -9.81, 0])
            self.scene.set_physics(self.physics)
        
        # UIの初期化
        if self.show_ui:
            self._setup_ui()
        
        # 更新スレッドの開始
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        # シーンの開始
        self.scene.start()
    
    def stop(self):
        """可視化の停止"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        if GENESIS_AVAILABLE and self.scene:
            self.scene.stop()
            self.scene = None
            self.model = None
            self.ui = None
            self.physics = None
    
    def queue_pose_update(self, data: Dict[str, Any]):
        """
        ポーズ更新のキューイング
        
        Args:
            data: ポーズデータ {
                "position": [x, y, z],
                "rotation": [roll, pitch, yaw],
                "joint_angles": [angle1, angle2, ...]
            }
        """
        self.update_queue.put(("pose", data))
    
    def queue_brain_state_update(self, data: Dict[str, Any]):
        """
        脳状態更新のキューイング
        
        Args:
            data: 脳状態データ {
                "neurotransmitter_levels": {
                    "dopamine": float,
                    "serotonin": float,
                    ...
                },
                "emotional_state": {
                    "primary_emotion": str,
                    "intensity": float
                }
            }
        """
        self.update_queue.put(("brain", data))
    
    def set_camera_view(self, view_mode: str):
        """
        カメラビューの設定
        
        Args:
            view_mode: カメラビューモード ('follow', 'fixed', 'first_person')
        """
        if view_mode in self.camera_settings:
            self.camera_view = view_mode
            if GENESIS_AVAILABLE and self.scene:
                self._setup_camera()
    
    def _setup_camera(self):
        """カメラのセットアップ"""
        if not GENESIS_AVAILABLE or not self.scene:
            return
        
        camera = gs.Camera()
        settings = self.camera_settings[self.camera_view]
        
        if self.camera_view == 'follow':
            camera.set_follow_mode(
                target=settings['target'],
                distance=settings['distance'],
                up=settings['up']
            )
        elif self.camera_view == 'fixed':
            camera.set_position(settings['position'])
            camera.set_target(settings['target'])
            camera.set_up(settings['up'])
        elif self.camera_view == 'first_person':
            camera.set_first_person_mode(
                position=settings['position'],
                forward=settings['forward'],
                up=settings['up']
            )
        
        camera.set_fov(settings.get('fov', 45.0))
        self.scene.set_camera(camera)
    
    def _setup_ui(self):
        """UIのセットアップ"""
        if not GENESIS_AVAILABLE or not self.scene:
            return
        
        self.ui = gs.UI()
        
        # 脳状態表示パネル
        self.ui.add_panel(
            title="脳状態",
            position=[10, 10],
            size=[200, 300]
        )
        
        # 神経伝達物質レベル表示
        self.ui.add_gauge(
            panel="脳状態",
            title="ドーパミン",
            min_value=0.0,
            max_value=1.0,
            initial_value=0.5
        )
        self.ui.add_gauge(
            panel="脳状態",
            title="セロトニン",
            min_value=0.0,
            max_value=1.0,
            initial_value=0.5
        )
        self.ui.add_gauge(
            panel="脳状態",
            title="ノルエピネフリン",
            min_value=0.0,
            max_value=1.0,
            initial_value=0.5
        )
        
        # 感情状態表示
        self.ui.add_text(
            panel="脳状態",
            title="感情状態",
            text="Neutral"
        )
        
        # コントロールパネル
        self.ui.add_panel(
            title="コントロール",
            position=[self.width - 210, 10],
            size=[200, 200]
        )
        
        # カメラ切り替えボタン
        self.ui.add_button(
            panel="コントロール",
            title="カメラ切替",
            callback=self._cycle_camera_view
        )
        
        # 物理シミュレーションの切り替え
        self.ui.add_checkbox(
            panel="コントロール",
            title="物理演算",
            checked=self.physics_enabled,
            callback=self._toggle_physics
        )
        
        self.scene.set_ui(self.ui)
    
    def _update_loop(self):
        """更新ループ"""
        while self.is_running:
            try:
                # キューからの更新を処理
                try:
                    update_type, data = self.update_queue.get(timeout=0.016)
                    
                    if update_type == "pose":
                        self._update_pose(data)
                    elif update_type == "brain":
                        self._update_brain_state(data)
                    
                    self.update_queue.task_done()
                except queue.Empty:
                    pass
                
                # 物理シミュレーションの更新（必要な場合）
                if GENESIS_AVAILABLE and self.scene and self.physics_enabled:
                    self.physics.step(0.016)
                
                time.sleep(0.001)  # CPUの過剰使用を防止
                
            except Exception as e:
                print(f"更新ループでエラーが発生しました: {e}")
    
    def _update_pose(self, data: Dict[str, Any]):
        """
        ポーズの更新
        
        Args:
            data: ポーズデータ
        """
        if not GENESIS_AVAILABLE or not self.model:
            return
        
        # 位置の設定
        if "position" in data:
            self.model.set_position(data["position"])
        
        # 回転の設定
        if "rotation" in data:
            self.model.set_rotation(data["rotation"])
        
        # 関節角度の設定
        if "joint_angles" in data:
            joint_angles = data["joint_angles"]
            for i, angle in enumerate(joint_angles):
                if i < self.model.get_joint_count():
                    self.model.set_joint_angle(i, angle)
    
    def _update_brain_state(self, data: Dict[str, Any]):
        """
        脳状態の更新
        
        Args:
            data: 脳状態データ
        """
        if not GENESIS_AVAILABLE or not self.ui:
            return
        
        # 神経伝達物質レベルの更新
        if "neurotransmitter_levels" in data:
            nt_levels = data["neurotransmitter_levels"]
            if "dopamine" in nt_levels:
                self.ui.update_gauge("脳状態", "ドーパミン", nt_levels["dopamine"])
            if "serotonin" in nt_levels:
                self.ui.update_gauge("脳状態", "セロトニン", nt_levels["serotonin"])
            if "norepinephrine" in nt_levels:
                self.ui.update_gauge("脳状態", "ノルエピネフリン", nt_levels["norepinephrine"])
        
        # 感情状態の更新
        if "emotional_state" in data:
            emotional_state = data["emotional_state"]
            emotion_text = f"{emotional_state['primary_emotion']} ({emotional_state['intensity']:.2f})"
            self.ui.update_text("脳状態", "感情状態", emotion_text)
    
    def _cycle_camera_view(self):
        """カメラビューの切り替え"""
        views = list(self.camera_settings.keys())
        current_index = views.index(self.camera_view)
        next_index = (current_index + 1) % len(views)
        self.set_camera_view(views[next_index])
    
    def _toggle_physics(self, enabled: bool):
        """
        物理シミュレーションの切り替え
        
        Args:
            enabled: 有効にするかどうか
        """
        self.physics_enabled = enabled
        
        if GENESIS_AVAILABLE and self.scene:
            if enabled:
                if not self.physics:
                    self.physics = gs.Physics()
                    self.physics.set_gravity([0, -9.81, 0])
                self.scene.set_physics(self.physics)
            else:
                self.scene.set_physics(None) 