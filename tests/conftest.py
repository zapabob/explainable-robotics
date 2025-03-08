"""
pytest構成ファイル
共通のテストフィクスチャやヘルパー関数を定義する
"""

import pytest
import sys
import os

# テスト用のフィクスチャを追加
@pytest.fixture
def sample_data():
    """テスト用のサンプルデータを提供するフィクスチャ"""
    return {
        "test_value": 42,
        "test_string": "explainable_robotics"
    } 