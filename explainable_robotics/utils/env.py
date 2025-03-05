"""
環境変数を管理するためのユーティリティモジュール。
.envファイルから環境変数を読み込む機能を提供します。
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env_file(env_file: Optional[Union[str, Path]] = None) -> None:
    """
    指定された.envファイルまたはデフォルトの場所から環境変数を読み込みます。
    
    Args:
        env_file: 環境変数ファイルのパス。Noneの場合、デフォルトの検索パスが使用されます。
    """
    if env_file is not None:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"環境変数を {env_path} から読み込みました")
        else:
            logger.warning(f"指定された環境変数ファイル {env_path} が見つかりませんでした")
    else:
        # デフォルトの場所から.envファイルを検索
        base_paths = [
            Path.cwd(),  # カレントディレクトリ
            Path.cwd().parent,  # 親ディレクトリ
            Path(__file__).parent.parent.parent,  # プロジェクトのルートディレクトリ
        ]
        
        for base_path in base_paths:
            env_path = base_path / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"環境変数を {env_path} から読み込みました")
                return
        
        logger.warning(".envファイルが見つかりませんでした。デフォルトの環境変数のみが使用されます。")


def get_env(key: str, default: Any = None) -> Any:
    """
    環境変数を取得します。
    
    Args:
        key: 環境変数のキー
        default: 環境変数が設定されていない場合のデフォルト値
        
    Returns:
        環境変数の値またはデフォルト値
    """
    return os.environ.get(key, default)


def get_required_env(key: str) -> str:
    """
    必須の環境変数を取得します。
    環境変数が見つからない場合は例外を発生させます。
    
    Args:
        key: 環境変数のキー
        
    Returns:
        環境変数の値
        
    Raises:
        ValueError: 環境変数が設定されていない場合
    """
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"必須の環境変数 {key} が設定されていません")
    return value


def get_api_key(service_name: str) -> str:
    """
    APIキーを取得します。
    
    Args:
        service_name: サービス名（例: "GEMINI", "OPENAI"）
        
    Returns:
        APIキー
        
    Raises:
        ValueError: APIキーが設定されていない場合
    """
    env_key = f"{service_name.upper()}_API_KEY"
    api_key = get_env(env_key)
    if api_key is None:
        raise ValueError(f"{service_name}のAPIキーが設定されていません。環境変数 {env_key} を設定してください。")
    return api_key


def get_all_env() -> Dict[str, str]:
    """
    すべての環境変数を取得します。
    
    Returns:
        すべての環境変数を含む辞書
    """
    return dict(os.environ) 