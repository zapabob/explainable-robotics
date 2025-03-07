"""
説明可視化ダッシュボードモジュール

このモジュールはロボットの行動説明を視覚化するためのダッシュボードを提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import datetime
import threading
import time

# 相対インポート
from ..utils.logging import get_logger
from ..visualization.activity_plots import (
    plot_layer_activity, 
    plot_motor_outputs, 
    plot_neurotransmitter_levels,
    plot_action_explanations,
    create_explanation_dashboard
)

# ロガーの設定
logger = get_logger(__name__)

class ExplanationDashboard:
    """
    ロボットの行動説明を視覚化するダッシュボード
    
    リアルタイムでロボットの行動履歴、神経活動、および説明を視覚化します。
    """
    
    def __init__(
        self,
        update_interval: float = 2.0,
        max_history_size: int = 100,
        save_dir: str = 'explanations',
        auto_save: bool = True
    ):
        """
        初期化
        
        Args:
            update_interval: 更新間隔（秒）
            max_history_size: 保持する履歴の最大サイズ
            save_dir: 保存ディレクトリ
            auto_save: 自動保存を有効にするかどうか
        """
        self.update_interval = update_interval
        self.max_history_size = max_history_size
        self.save_dir = save_dir
        self.auto_save = auto_save
        
        # データストレージ
        self.motor_history: List[np.ndarray] = []
        self.explanation_history: List[Dict[str, Any]] = []
        self.nt_history: List[Dict[str, float]] = []
        self.layer_activities: Dict[str, np.ndarray] = {}
        self.model_info: Dict[str, Any] = {}
        
        # 実行状態
        self.running = False
        self.update_thread = None
        
        # 保存ディレクトリの作成
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 読み込み中フラグ
        self.loading = False
        
        logger.info(f"説明ダッシュボードを初期化: 更新間隔={update_interval}秒")
    
    def start(self):
        """
        更新ループを開始
        """
        if self.running:
            logger.warning("ダッシュボードは既に実行中です")
            return
        
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("ダッシュボード更新ループを開始しました")
    
    def stop(self):
        """
        更新ループを停止
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None
        
        logger.info("ダッシュボード更新ループを停止しました")
    
    def _update_loop(self):
        """
        メイン更新ループ
        """
        last_save = time.time()
        
        while self.running:
            try:
                # データの読み込み中は更新をスキップ
                if self.loading:
                    time.sleep(0.5)
                    continue
                
                # ダッシュボードを更新
                self._update_dashboard()
                
                # 自動保存
                if self.auto_save and (time.time() - last_save) > 60:  # 1分ごとに保存
                    self.save_dashboard()
                    last_save = time.time()
                
                # 更新間隔で待機
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"ダッシュボード更新エラー: {str(e)}")
                time.sleep(5.0)  # エラー時は長めに待機
    
    def _update_dashboard(self):
        """
        ダッシュボードを更新
        """
        # 十分なデータがある場合のみ更新
        if not self.motor_history or not self.explanation_history:
            return
        
        # ダッシュボードを作成
        try:
            action_history = [
                {
                    'value': motor.tolist() if isinstance(motor, np.ndarray) else motor,
                    'explanation': exp,
                    'context': {'neurotransmitter_levels': self.nt_history[i]} if i < len(self.nt_history) else {}
                }
                for i, (motor, exp) in enumerate(zip(self.motor_history, self.explanation_history))
            ]
            
            fig, save_path = create_explanation_dashboard(
                model_info=self.model_info,
                action_history=action_history,
                layer_activities=self.layer_activities,
                save_dir=None  # 自動保存は行わない
            )
            
            # 表示
            plt.pause(0.01)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"ダッシュボード作成エラー: {str(e)}")
    
    def update_motor_command(self, motor_command: Union[np.ndarray, List[float]]):
        """
        モーターコマンドを更新
        
        Args:
            motor_command: モーターコマンド値
        """
        # NumPy配列に変換
        if not isinstance(motor_command, np.ndarray):
            motor_command = np.array(motor_command)
        
        # 履歴に追加
        self.motor_history.append(motor_command)
        
        # 履歴サイズの制限
        if len(self.motor_history) > self.max_history_size:
            self.motor_history = self.motor_history[-self.max_history_size:]
    
    def update_explanation(self, explanation: Dict[str, Any]):
        """
        説明を更新
        
        Args:
            explanation: 説明情報
        """
        # 履歴に追加
        self.explanation_history.append(explanation)
        
        # nt_levelsも更新（含まれている場合）
        if 'neurotransmitter_levels' in explanation:
            self.nt_history.append(explanation['neurotransmitter_levels'])
        
        # 履歴サイズの制限
        if len(self.explanation_history) > self.max_history_size:
            self.explanation_history = self.explanation_history[-self.max_history_size:]
        
        if len(self.nt_history) > self.max_history_size:
            self.nt_history = self.nt_history[-self.max_history_size:]
    
    def update_layer_activities(self, layer_activities: Dict[str, np.ndarray]):
        """
        層活性化を更新
        
        Args:
            layer_activities: 層名と活性化値の辞書
        """
        self.layer_activities = {k: np.array(v) for k, v in layer_activities.items()}
    
    def update_model_info(self, model_info: Dict[str, Any]):
        """
        モデル情報を更新
        
        Args:
            model_info: モデル情報の辞書
        """
        self.model_info = model_info.copy()
    
    def save_dashboard(self, timestamp: Optional[str] = None) -> str:
        """
        ダッシュボードを保存
        
        Args:
            timestamp: タイムスタンプ（Noneの場合は現在時刻）
            
        Returns:
            保存先のパス
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # データの準備
        action_history = [
            {
                'value': motor.tolist() if isinstance(motor, np.ndarray) else motor,
                'explanation': exp,
                'context': {'neurotransmitter_levels': self.nt_history[i]} if i < len(self.nt_history) else {}
            }
            for i, (motor, exp) in enumerate(zip(self.motor_history, self.explanation_history))
        ]
        
        # ダッシュボードを作成して保存
        fig, save_path = create_explanation_dashboard(
            model_info=self.model_info,
            action_history=action_history,
            layer_activities=self.layer_activities,
            save_dir=self.save_dir
        )
        
        plt.close(fig)
        
        # 全データもJSONとして保存
        data_path = f"{self.save_dir}/dashboard_data_{timestamp}.json"
        data = {
            'model_info': self.model_info,
            'explanations': self.explanation_history,
            'neurotransmitter_history': self.nt_history,
            'timestamp': timestamp
        }
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"ダッシュボードを保存しました: {save_path}")
        return save_path
    
    def load_dashboard_data(self, filepath: str):
        """
        保存されたダッシュボードデータを読み込み
        
        Args:
            filepath: ダッシュボードデータのJSONファイルパス
        """
        try:
            self.loading = True
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # データを読み込み
            self.model_info = data.get('model_info', {})
            self.explanation_history = data.get('explanations', [])
            self.nt_history = data.get('neurotransmitter_history', [])
            
            # モーターコマンドの再構築
            self.motor_history = []
            for exp in self.explanation_history:
                if 'motor_actions' in exp:
                    actions = exp['motor_actions']
                    # 辞書からリストに変換
                    motor_list = [actions.get(f'motor_{i}', 0.0) for i in range(10)]
                    self.motor_history.append(np.array(motor_list))
            
            logger.info(f"ダッシュボードデータを読み込みました: {filepath}")
            
        except Exception as e:
            logger.error(f"ダッシュボードデータ読み込みエラー: {str(e)}")
        
        finally:
            self.loading = False
    
    def create_static_dashboard(
        self,
        title: str = "行動説明ダッシュボード",
        fig_size: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        静的なダッシュボードを作成（単一の図として）
        
        Args:
            title: ダッシュボードのタイトル
            fig_size: 図のサイズ
            save_path: 保存先のパス（Noneの場合は保存しない）
            show: プロットを表示するかどうか
            
        Returns:
            matplotlib図オブジェクト
        """
        # データの準備
        action_history = [
            {
                'value': motor.tolist() if isinstance(motor, np.ndarray) else motor,
                'explanation': exp,
                'context': {'neurotransmitter_levels': self.nt_history[i]} if i < len(self.nt_history) else {}
            }
            for i, (motor, exp) in enumerate(zip(self.motor_history, self.explanation_history))
        ]
        
        # 指定された引数でダッシュボードを作成
        fig, _ = create_explanation_dashboard(
            model_info=self.model_info,
            action_history=action_history,
            layer_activities=self.layer_activities,
            save_dir=None
        )
        
        # タイトルを設定
        fig.suptitle(title, fontsize=16)
        
        # 保存
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 表示
        if show:
            plt.show()
        
        return fig 