"""
ExplainableRobotics - 神経科学的に妥当なヒューマノイドロボット制御のためのフレームワーク
"""

import os
import sys
import warnings
import importlib.metadata
from typing import Dict, Tuple, TypeAlias, Any, List, Optional, Union, Literal

# Python バージョンチェック
if not (sys.version_info.major == 3 and sys.version_info.minor >= 12):
    warnings.warn(
        f"Python 3.12以上を推奨します。現在のバージョン: {sys.version_info.major}.{sys.version_info.minor}",
        RuntimeWarning
    )

__version__ = "0.1.0"
__author__ = "ExplainableRobotics Team"
__license__ = "MIT"
__copyright__ = "Copyright 2023-2024 Explainable Robotics Team"

# Python 3.13以降のための型エイリアス（PEP 695対応）
# Python 3.12以前でも動作するようにTypeAliasを使用
JsonDict: TypeAlias = Dict[str, Any]
NeurotransmitterLevels: TypeAlias = Dict[str, float]
SensorData: TypeAlias = Dict[str, Any]
LLMProvider: TypeAlias = Literal["openai", "claude", "gemini", "vertex", "local"]

# コア機能のインポート
from explainable_robotics.core.integrated_system import create_integrated_system, IntegratedSystem

# バージョン情報
VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "description": "初期リリース - 神経伝達物質モデル、大脳皮質モデル、ロボットインターフェース、LLM統合",
    "python_compat": ["3.12", "3.13", "3.14"]
}

def get_version() -> str:
    """バージョン文字列を返す"""
    return __version__

def get_version_info() -> Dict[str, Any]:
    """バージョン情報を辞書として返す"""
    return VERSION_INFO.copy()

# サブパッケージのインポート
try:
    from . import cortical
    from . import core
    from . import visualization
    from . import controller
    from . import utils
    from . import demos

    # 便利な関数のエクスポート
    from .cortical import BioKAN
    from .core import multi_llm_agent
    from .core.multi_llm_agent import MultiLLMAgent
    from .visualization import GenesisVisualizer
    from .controller import RobotController
except ImportError as e:
    warnings.warn(
        f"モジュールのインポートエラー: {e}。一部の機能が利用できない可能性があります。",
        ImportWarning
    )

# セットアップが完了しているかどうかを確認
_SETUP_DONE = False

def check_dependencies() -> Tuple[bool, Dict[str, bool]]:
    """
    依存関係のチェック
    
    Returns:
        成功かどうかとモジュールの利用可否の辞書
    """
    # 必須の依存関係
    required_dependencies = ["torch", "numpy", "colorlog", "pyyaml"]
    
    # オプションの依存関係
    optional_dependencies = [
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_anthropic",
        "langchain_google_genai",
        "langchain_google_vertexai",
        "google.generativeai",
        "openai",
        "anthropic",
        "huggingface_hub",
        "transformers",
        "genesis_world",
        "cryptography",
    ]
    
    # 依存関係のチェック
    dependencies = {}
    
    # 必須の依存関係のチェック
    required_ok = True
    for dep in required_dependencies:
        try:
            __import__(dep.split(".")[0])
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            required_ok = False
    
    # オプションの依存関係のチェック
    for dep in optional_dependencies:
        try:
            __import__(dep.split(".")[0])
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    return required_ok, dependencies

def get_installed_packages() -> Dict[str, str]:
    """
    インストールされたパッケージとバージョンの一覧を取得
    
    Returns:
        パッケージ名とバージョンの辞書
    """
    packages = {}
    try:
        installed_packages = importlib.metadata.distributions()
        for pkg in installed_packages:
            packages[pkg.metadata["Name"]] = pkg.metadata["Version"]
    except Exception as e:
        warnings.warn(f"パッケージ情報の取得に失敗しました: {e}", RuntimeWarning)
    
    return packages

def setup():
    """
    ライブラリのセットアップ
    
    追加のパスの設定や初期化を行います。
    """
    global _SETUP_DONE
    
    if _SETUP_DONE:
        return
    
    # モデルディレクトリの追加
    module_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(os.path.dirname(module_path), "models")
    data_path = os.path.join(os.path.dirname(module_path), "data")
    logs_path = os.path.join(os.path.dirname(module_path), "logs")
    
    # 必要なディレクトリが存在しない場合は作成
    for path in [models_path, data_path, logs_path]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    
    # 依存関係のチェック
    required_ok, dependencies = check_dependencies()
    
    if not required_ok:
        missing = [dep for dep, available in dependencies.items() 
                  if not available and dep in ["torch", "numpy", "colorlog", "pyyaml"]]
        warnings.warn(
            f"警告: 必須の依存関係が不足しています: {', '.join(missing)}。\n"
            f"pip install explainable-robotics[all] を実行して全ての依存関係をインストールしてください。",
            RuntimeWarning
        )
    
    # Python 3.13以降の機能があるかチェック
    has_enhanced_typing = False
    try:
        from typing import TypeAlias, assert_type, reveal_type
        has_enhanced_typing = True
    except ImportError:
        pass
    
    if has_enhanced_typing and sys.version_info >= (3, 13):
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} で実行中 - 拡張型ヒント機能が利用可能です。")
    
    _SETUP_DONE = True

# 自動セットアップ
setup() 