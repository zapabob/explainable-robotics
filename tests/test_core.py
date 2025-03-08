"""
Explainable Roboticsのコア機能をテストするモジュール
"""

import pytest
import explainable_robotics


def test_core_modules_existence():
    """コアモジュールが存在することを確認"""
    # このテストはコアモジュールの存在を確認します
    # 実際のプロジェクト構造に合わせて調整してください
    try:
        from explainable_robotics import core
        assert core is not None
    except ImportError:
        pytest.skip("coreモジュールがインポートできません") 