"""
神経活動とモーター出力の視覚化モジュール

このモジュールは大脳皮質モデルの層活性化とロボットの行動を視覚化するための関数を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Union, Any
import os
import datetime
import json

def plot_layer_activity(
    layer_activities: Dict[str, np.ndarray],
    fig_size: Tuple[int, int] = (12, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    大脳皮質の層活性化を視覚化
    
    Args:
        layer_activities: 層名とその活性化値を含む辞書
        fig_size: 図のサイズ
        cmap: カラーマップ
        save_path: 保存先のパス（Noneの場合は保存しない）
        show: プロットを表示するかどうか
        title: プロットのタイトル
        
    Returns:
        matplotlib図オブジェクト
    """
    num_layers = len(layer_activities)
    
    # 図を作成
    fig = plt.figure(figsize=fig_size)
    
    # グリッドを設定
    if num_layers <= 3:
        grid = gridspec.GridSpec(1, num_layers)
    else:
        grid = gridspec.GridSpec(2, (num_layers + 1) // 2)
    
    # 各層のプロット
    for i, (layer_name, activity) in enumerate(layer_activities.items()):
        # 2次元以上の場合は最初の2次元のみを使用
        if activity.ndim > 2:
            activity = activity.reshape(activity.shape[0], -1)
        
        # 1次元の場合は2次元に変換
        if activity.ndim == 1:
            # 正方形に近い2次元配列に変換
            side = int(np.sqrt(activity.shape[0]))
            activity = activity[:side*side].reshape(side, side)
        
        # サブプロットを作成
        row = i // ((num_layers + 1) // 2) if num_layers > 3 else 0
        col = i % ((num_layers + 1) // 2) if num_layers > 3 else i
        
        ax = plt.subplot(grid[row, col])
        im = ax.imshow(activity, cmap=cmap, aspect='auto')
        ax.set_title(f"層 {layer_name}")
        plt.colorbar(im, ax=ax)
    
    # 全体のタイトル
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    
    return fig

def plot_motor_outputs(
    motor_history: List[np.ndarray],
    joint_names: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    モーター出力の履歴を視覚化
    
    Args:
        motor_history: モーター出力履歴の配列のリスト
        joint_names: 関節名のリスト（Noneの場合は数字で表示）
        fig_size: 図のサイズ
        save_path: 保存先のパス（Noneの場合は保存しない）
        show: プロットを表示するかどうか
        title: プロットのタイトル
        
    Returns:
        matplotlib図オブジェクト
    """
    # モーター出力を行列に変換
    motor_array = np.array(motor_history)
    num_timesteps, num_motors = motor_array.shape
    
    # 関節名が指定されていない場合は生成
    if joint_names is None:
        joint_names = [f"関節{i+1}" for i in range(num_motors)]
    
    # 図を作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    # 時系列プロット
    for i in range(min(num_motors, 10)):  # 最大10関節まで表示
        ax1.plot(motor_array[:, i], label=joint_names[i])
    
    ax1.set_xlabel('時間ステップ')
    ax1.set_ylabel('モーター出力 (-1〜1)')
    ax1.set_title('モーター出力の時系列')
    ax1.grid(True)
    ax1.legend()
    
    # ヒートマップ
    im = ax2.imshow(motor_array.T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xlabel('時間ステップ')
    ax2.set_ylabel('モーター')
    ax2.set_title('モーター出力のヒートマップ')
    plt.colorbar(im, ax=ax2)
    
    # y軸ラベルを設定（モーター名）
    if num_motors <= 20:  # 表示するモーターが少ない場合のみラベルを表示
        ax2.set_yticks(np.arange(num_motors))
        ax2.set_yticklabels(joint_names)
    
    # 全体のタイトル
    if title:
        plt.suptitle(title)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    
    return fig

def plot_neurotransmitter_levels(
    history: List[Dict[str, float]],
    fig_size: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    神経伝達物質レベルの履歴を視覚化
    
    Args:
        history: 神経伝達物質レベルの履歴辞書のリスト
        fig_size: 図のサイズ
        save_path: 保存先のパス（Noneの場合は保存しない）
        show: プロットを表示するかどうか
        
    Returns:
        matplotlib図オブジェクト
    """
    # 神経伝達物質名を取得
    nt_names = set()
    for entry in history:
        nt_names.update(entry.keys())
    nt_names = sorted(list(nt_names))
    
    # データを整形
    data = {name: [] for name in nt_names}
    for entry in history:
        for name in nt_names:
            data[name].append(entry.get(name, 0.0))
    
    # 図を作成
    fig, ax = plt.subplots(figsize=fig_size)
    
    # 各神経伝達物質をプロット
    for name in nt_names:
        ax.plot(data[name], label=name)
    
    # ラベルと凡例
    ax.set_xlabel('時間ステップ')
    ax.set_ylabel('神経伝達物質レベル (0〜1)')
    ax.set_title('神経伝達物質レベルの変化')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    
    return fig

def plot_3d_robot_movement(
    positions: List[np.ndarray],
    fig_size: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """
    ロボットの3D動作軌跡を視覚化
    
    Args:
        positions: ロボット位置の配列のリスト（各要素はx,y,z座標）
        fig_size: 図のサイズ
        save_path: 保存先のパス（Noneの場合は保存しない）
        show: プロットを表示するかどうか
        title: プロットのタイトル
        
    Returns:
        matplotlib図オブジェクト
    """
    # 位置データを配列に変換
    pos_array = np.array(positions)
    
    # 図を作成
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    
    # 軌跡をプロット
    ax.plot3D(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'b-')
    
    # 始点と終点をマーク
    ax.scatter(pos_array[0, 0], pos_array[0, 1], pos_array[0, 2], c='g', marker='o', s=100, label='開始')
    ax.scatter(pos_array[-1, 0], pos_array[-1, 1], pos_array[-1, 2], c='r', marker='x', s=100, label='終了')
    
    # ラベルと凡例
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title or 'ロボットの動作軌跡')
    ax.legend()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    
    return fig

def plot_action_explanations(
    explanations: List[Dict[str, Any]],
    max_actions: int = 10,
    fig_size: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    行動の説明を視覚化
    
    Args:
        explanations: 説明情報を含む辞書のリスト
        max_actions: 表示する行動の最大数
        fig_size: 図のサイズ
        save_path: 保存先のパス（Noneの場合は保存しない）
        show: プロットを表示するかどうか
        
    Returns:
        matplotlib図オブジェクト
    """
    # 表示する行動を制限
    explanations = explanations[-max_actions:] if len(explanations) > max_actions else explanations
    
    # 図を作成
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1])
    
    # 確信度のプロット
    confidences = [exp.get('confidence', 0.0) for exp in explanations]
    ax1 = plt.subplot(gs[0])
    ax1.plot(confidences, 'b-o')
    ax1.set_xlabel('行動インデックス')
    ax1.set_ylabel('確信度')
    ax1.set_title('行動確信度')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    # モーター出力のプロット
    motor_actions = []
    for exp in explanations:
        actions = exp.get('motor_actions', {})
        # 辞書を配列に変換
        action_array = np.array([actions.get(f'motor_{i}', 0.0) for i in range(10)])
        motor_actions.append(action_array)
    
    motor_array = np.array(motor_actions)
    ax2 = plt.subplot(gs[1])
    im = ax2.imshow(motor_array, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xlabel('モーター')
    ax2.set_ylabel('行動インデックス')
    ax2.set_title('モーター出力')
    plt.colorbar(im, ax=ax2)
    
    # 説明テキスト
    ax3 = plt.subplot(gs[2])
    ax3.axis('off')
    
    # 最新の説明テキストを表示
    if explanations:
        latest_exp = explanations[-1]
        narrative = latest_exp.get('narrative', '説明なし')
        reasoning = latest_exp.get('reasoning', '理由なし')
        
        explanation_text = f"最新の行動説明:\n{narrative}\n\n理由:\n{reasoning}"
        ax3.text(0.05, 0.5, explanation_text, verticalalignment='center', wrap=True)
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 表示
    if show:
        plt.show()
    
    return fig

def create_explanation_dashboard(
    model_info: Dict[str, Any],
    action_history: List[Dict[str, Any]],
    layer_activities: Dict[str, np.ndarray],
    save_dir: Optional[str] = None
) -> Tuple[plt.Figure, str]:
    """
    説明ダッシュボードを作成
    
    Args:
        model_info: モデル情報
        action_history: 行動履歴
        layer_activities: 層活性化
        save_dir: 保存ディレクトリ（Noneの場合は保存しない）
        
    Returns:
        (matplotlib図オブジェクト, 保存パス)
    """
    # タイムスタンプ
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # モデル情報
    model_type = model_info.get('model_type', '不明')
    uses_glia = model_info.get('uses_glia', False)
    uses_neuromodulation = model_info.get('uses_neuromodulation', False)
    
    # 行動履歴の処理
    motor_commands = []
    explanations = []
    nt_levels = []
    
    for action in action_history:
        if 'value' in action:
            motor_commands.append(np.array(action['value']))
        if 'explanation' in action:
            explanations.append(action['explanation'])
        if 'context' in action and 'neurotransmitter_levels' in action['context']:
            nt_levels.append(action['context']['neurotransmitter_levels'])
    
    # 図を作成
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2)
    
    # 層活性化のプロット
    ax1 = plt.subplot(gs[0, 0])
    for layer_name, activity in layer_activities.items():
        if activity.ndim > 2:
            activity = activity.reshape(activity.shape[0], -1)
        if activity.ndim == 1:
            side = int(np.sqrt(activity.shape[0]))
            activity = activity[:side*side].reshape(side, side)
        im = ax1.imshow(activity, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        break  # 最初の層のみを表示
    ax1.set_title(f"層活性化（{list(layer_activities.keys())[0]}）")
    
    # モーター出力のプロット
    if motor_commands:
        motor_array = np.array(motor_commands)
        ax2 = plt.subplot(gs[0, 1])
        im = ax2.imshow(motor_array[-20:].T, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_xlabel('時間ステップ')
        ax2.set_ylabel('モーター')
        ax2.set_title('最近のモーター出力')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 神経伝達物質レベルのプロット
    if nt_levels:
        ax3 = plt.subplot(gs[1, 0])
        nt_names = list(nt_levels[0].keys())
        
        for name in nt_names:
            values = [entry.get(name, 0.0) for entry in nt_levels]
            ax3.plot(values[-50:], label=name)  # 最新の50エントリ
        
        ax3.set_xlabel('時間ステップ')
        ax3.set_ylabel('レベル')
        ax3.set_title('神経伝達物質レベル')
        ax3.set_ylim(0, 1)
        ax3.grid(True)
        ax3.legend()
    
    # 最新の説明
    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    
    if explanations:
        latest_exp = explanations[-1]
        narrative = latest_exp.get('narrative', '説明なし')
        reasoning = latest_exp.get('reasoning', '理由なし')
        confidence = latest_exp.get('confidence', 0.0)
        
        explanation_text = f"行動説明:\n{narrative}\n\n理由:\n{reasoning}\n\n確信度: {confidence:.1%}"
        ax4.text(0.05, 0.5, explanation_text, verticalalignment='center', wrap=True)
    
    # モデル情報のテキスト
    ax5 = plt.subplot(gs[2, :])
    ax5.axis('off')
    
    info_text = (
        f"モデル情報:\n"
        f"タイプ: {model_type}\n"
        f"グリア細胞: {'有効' if uses_glia else '無効'}\n"
        f"神経調節: {'有効' if uses_neuromodulation else '無効'}\n"
        f"行動履歴数: {len(action_history)}\n"
        f"最終更新: {timestamp}"
    )
    ax5.text(0.05, 0.5, info_text, verticalalignment='center')
    
    plt.suptitle("神経活動と行動の説明ダッシュボード", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    save_path = None
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}/dashboard_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 説明をJSONとしても保存
        json_path = f"{save_dir}/explanation_{timestamp}.json"
        if explanations:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(explanations[-1], f, indent=2, ensure_ascii=False)
    
    return fig, save_path 