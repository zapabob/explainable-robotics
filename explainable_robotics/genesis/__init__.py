"""
Genesisライブラリとの統合モジュール

実際のGenesisライブラリがインストールされていない場合は、
モック実装が自動的に使用されます。
"""

# genesisモジュールをインポートまたはモック化
GENESIS_AVAILABLE = False

# ダミーモジュールのベースクラス
class DummyModule:
    """任意の属性呼び出しに対してダミーオブジェクトを返すクラス"""
    def __getattr__(self, name):
        """存在しない属性アクセスに対してダミーモジュールを返す"""
        return DummyModule()
    
    def __call__(self, *args, **kwargs):
        """関数として呼び出された場合も自身を返す"""
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

# インターフェースのエクスポート
from .robot_interface import GenesisRobotInterface, create_robot_interface

__all__ = [
    'GenesisRobotInterface',
    'create_robot_interface',
    'GENESIS_AVAILABLE',
    'gs'
] 