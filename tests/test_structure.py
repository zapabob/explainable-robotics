"""
プロジェクト構造をテストするモジュール
"""

import os
import pytest


def test_main_package_exists():
    """メインパッケージが存在することを確認"""
    assert os.path.isdir("explainable_robotics")


def test_tests_directory_exists():
    """テストディレクトリが存在することを確認"""
    assert os.path.isdir("tests")


def test_pyproject_toml_exists():
    """pyproject.tomlファイルが存在することを確認"""
    assert os.path.isfile("pyproject.toml") 