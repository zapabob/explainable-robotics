"""
Genesisライブラリとの統合モジュール。

実際のGenesisライブラリがインストールされていない場合は、
モック実装が自動的に使用されます。
"""

# genesisモジュールをインポートまたはモック化
GENESIS_AVAILABLE = False

# ダミーモジュールのベースクラス
class DummyModule:
    def __getattr__(self, name):
        return DummyModule()
    
    def __call__(self, *args, **kwargs):
        return self

try:
    # 実際のgenesisライブラリを試みる
    import genesis as gs
    if hasattr(gs, 'humanoid'):
        GENESIS_AVAILABLE = True
    else:
        # 構造が不完全な場合
        gs = DummyModule()
        print("WARNING: Genesisライブラリ構造が不完全です。モック実装を使用します。")
except ImportError:
    # モジュールが見つからない場合
    gs = DummyModule()
    print("WARNING: Genesisライブラリが見つかりません。モック実装を使用します。")

# 便利なクラスを直接エクスポート
from . import robot_interface
from .robot_interface import (
    GenesisRobotInterface,
    create_robot_interface
)

# 必要なクラスや関数をエクスポート
__all__ = [
    'GenesisRobotInterface',
    'create_robot_interface',
    'GENESIS_AVAILABLE',
    'gs',  # genesis as gsの形式でインポートするためにモジュール自体をエクスポート
    'robot_interface'
] 