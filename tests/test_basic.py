"""
基本的なテスト機能を確認するためのテストモジュール
"""

import pytest
import explainable_robotics


def test_version():
    """バージョン情報が文字列として存在することを確認"""
    assert isinstance(explainable_robotics.__version__, str)
    assert len(explainable_robotics.__version__) > 0


def test_package_import():
    """パッケージが正しくインポートできることを確認"""
    assert explainable_robotics is not None 