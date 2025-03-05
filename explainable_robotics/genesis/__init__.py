"""
Genesis模擬ライブラリ

ヒューマノイドロボットのシミュレーションと可視化のためのライブラリです。
"""

# 実際のGensisライブラリが存在するか確認
try:
    import genesis as real_genesis
    # 実際のライブラリが存在する場合は、それをそのまま使用
    from genesis import *
    USING_REAL_GENESIS = True
    
except ImportError:
    # 実際のライブラリがない場合は、モックを使用
    from .mock_genesis import *
    from .mock_genesis import Environment, HumanoidRobot, Viewer, NeurotransmitterSystem
    USING_REAL_GENESIS = False
    
    # 名前空間にクラスを追加
    __all__ = ['Environment', 'HumanoidRobot', 'Viewer', 'NeurotransmitterSystem',
               'visualization', 'robot', 'motor', 'neurotransmitters'] 