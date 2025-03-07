#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
genesisのインポートテストスクリプト
explainable-roboticsパッケージが正しく設定されていることを確認するためのテスト
"""

import sys
import os

# インポートテスト
print("Genesisインポートテスト:")
print("="*50)

# 1. まずpythonのバージョンを確認
print(f"Python バージョン: {sys.version}")

# 2. 直接genesisをインポート
try:
    import genesis as gs
    print("✓ import genesis as gs: 成功")
    print(f"  - モジュールパス: {gs.__file__ if hasattr(gs, '__file__') else '不明'}")
    
    # サブモジュールのチェック
    if hasattr(gs, 'humanoid'):
        print("  - gs.humanoid: 存在します")
    else:
        print("  - gs.humanoid: 存在しません")
        
except ImportError as e:
    print(f"✗ import genesis as gs: 失敗 - {e}")

# 3. explainable_roboticsを通してgenesisをインポート
print("\nExplainable Roboticsからのインポート:")
try:
    import explainable_robotics
    print(f"✓ explainable_robotics インポート成功: v{explainable_robotics.__version__}")
    
    # explainable_robotics.genesis.gsをインポート
    try:
        from explainable_robotics.genesis import gs as er_gs
        print("✓ from explainable_robotics.genesis import gs: 成功")
        
        # 同じオブジェクトかチェック
        try:
            import genesis as original_gs
            print(f"  - 同じオブジェクト: {original_gs is er_gs}")
        except ImportError:
            print("  - オリジナルのgenesisモジュールは利用できません")
            
    except ImportError as e:
        print(f"✗ from explainable_robotics.genesis import gs: 失敗 - {e}")
    
    # GenesisRobotInterfaceのインポート
    try:
        from explainable_robotics.genesis import GenesisRobotInterface
        print("✓ GenesisRobotInterface: インポート成功")
        print(f"  - 利用可能: {explainable_robotics.genesis.GENESIS_AVAILABLE}")
    except ImportError as e:
        print(f"✗ GenesisRobotInterface: インポート失敗 - {e}")
        
except ImportError as e:
    print(f"✗ explainable_robotics インポート失敗: {e}")

print("\nテスト完了")
print("="*50) 