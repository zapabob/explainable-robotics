"""セキュリティユーティリティ

APIキーやその他の機密情報を安全に管理するためのユーティリティモジュール。
環境変数、暗号化ファイル、キーストアからの読み込みをサポートします。
"""

import os
import json
import base64
import getpass
import hashlib
import secrets
from typing import Dict, Any, Optional, Union
from pathlib import Path

# 暗号化ライブラリが利用可能な場合は使用
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    print("WARNING: cryptographyライブラリが利用できません。暗号化機能は制限されます。")

# ロギングの設定
from .logging import get_logger
logger = get_logger(__name__)


class SecureKeyManager:
    """
    セキュアなAPIキー管理クラス
    
    APIキーや機密情報を安全に管理するためのクラスです。
    環境変数、暗号化ファイル、またはキーストアからキーを取得できます。
    """
    
    def __init__(
        self,
        keys_file: Optional[str] = None,
        use_keyring: bool = True,
        master_password: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            keys_file: キーファイルのパス（デフォルト: ~/.explainable_robotics/keys.json）
            use_keyring: システムキーリングを使用するかどうか
            master_password: マスターパスワード（Noneの場合は必要時に要求）
        """
        self.use_keyring = use_keyring
        self.master_password = master_password
        
        # キーリングのインポート試行
        self.keyring_available = False
        if use_keyring:
            try:
                import keyring
                self.keyring = keyring
                self.keyring_available = True
            except ImportError:
                logger.warning("keyringライブラリが利用できません。システムキーリングは使用されません。")
        
        # デフォルトのキーファイルパスの設定
        if keys_file is None:
            home_dir = str(Path.home())
            config_dir = os.path.join(home_dir, '.explainable_robotics')
            os.makedirs(config_dir, exist_ok=True)
            self.keys_file = os.path.join(config_dir, 'keys.json')
        else:
            self.keys_file = keys_file
            os.makedirs(os.path.dirname(keys_file), exist_ok=True)
        
        # 暗号化キーの初期化
        self.encryption_key = None
        
        # キャッシュの初期化
        self.cache = {}
        
        logger.debug(f"セキュリティマネージャーを初期化しました（キーファイル: {self.keys_file}）")
    
    def get_key(self, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        キーの取得
        
        以下の順序でキーを検索します:
        1. キャッシュ
        2. 環境変数
        3. キーリング（有効な場合）
        4. 暗号化されたキーファイル
        
        Args:
            key_name: キー名
            default: デフォルト値（キーが見つからない場合）
        
        Returns:
            キー値またはデフォルト値
        """
        # キャッシュをチェック
        if key_name in self.cache:
            return self.cache[key_name]
        
        # 環境変数をチェック
        env_value = os.environ.get(key_name)
        if env_value:
            self.cache[key_name] = env_value
            return env_value
        
        # キーリングをチェック
        if self.keyring_available:
            try:
                keyring_value = self.keyring.get_password("explainable_robotics", key_name)
                if keyring_value:
                    self.cache[key_name] = keyring_value
                    return keyring_value
            except Exception as e:
                logger.warning(f"キーリングからのキー取得に失敗しました: {e}")
        
        # 暗号化されたキーファイルをチェック
        try:
            keys = self._load_keys_file()
            if key_name in keys:
                key_value = keys[key_name]
                self.cache[key_name] = key_value
                return key_value
        except Exception as e:
            logger.warning(f"キーファイルからのキー取得に失敗しました: {e}")
        
        # デフォルト値を返す
        return default
    
    def set_key(self, key_name: str, key_value: str, save_to_keyring: bool = True, save_to_file: bool = True) -> bool:
        """
        キーの設定
        
        Args:
            key_name: キー名
            key_value: キー値
            save_to_keyring: キーリングに保存するかどうか
            save_to_file: キーファイルに保存するかどうか
        
        Returns:
            成功したかどうか
        """
        # キャッシュに保存
        self.cache[key_name] = key_value
        
        success = True
        
        # キーリングに保存
        if save_to_keyring and self.keyring_available:
            try:
                self.keyring.set_password("explainable_robotics", key_name, key_value)
                logger.debug(f"キー '{key_name}' をキーリングに保存しました")
            except Exception as e:
                logger.error(f"キーリングへのキー保存に失敗しました: {e}")
                success = False
        
        # キーファイルに保存
        if save_to_file:
            try:
                keys = self._load_keys_file()
                keys[key_name] = key_value
                self._save_keys_file(keys)
                logger.debug(f"キー '{key_name}' をキーファイルに保存しました")
            except Exception as e:
                logger.error(f"キーファイルへのキー保存に失敗しました: {e}")
                success = False
        
        return success
    
    def delete_key(self, key_name: str, delete_from_keyring: bool = True, delete_from_file: bool = True) -> bool:
        """
        キーの削除
        
        Args:
            key_name: キー名
            delete_from_keyring: キーリングから削除するかどうか
            delete_from_file: キーファイルから削除するかどうか
        
        Returns:
            成功したかどうか
        """
        # キャッシュから削除
        if key_name in self.cache:
            del self.cache[key_name]
        
        success = True
        
        # キーリングから削除
        if delete_from_keyring and self.keyring_available:
            try:
                self.keyring.delete_password("explainable_robotics", key_name)
                logger.debug(f"キー '{key_name}' をキーリングから削除しました")
            except Exception as e:
                logger.warning(f"キーリングからのキー削除に失敗しました: {e}")
                success = False
        
        # キーファイルから削除
        if delete_from_file:
            try:
                keys = self._load_keys_file()
                if key_name in keys:
                    del keys[key_name]
                    self._save_keys_file(keys)
                    logger.debug(f"キー '{key_name}' をキーファイルから削除しました")
            except Exception as e:
                logger.error(f"キーファイルからのキー削除に失敗しました: {e}")
                success = False
        
        return success
    
    def list_keys(self) -> Dict[str, str]:
        """
        利用可能なキーの一覧を取得
        
        Returns:
            キー名とその値の辞書
        """
        keys = {}
        
        # キャッシュからのキー
        keys.update(self.cache)
        
        # 環境変数からのキー（APIまたはKEYを含む変数のみ）
        for env_name, env_value in os.environ.items():
            if 'API' in env_name or 'KEY' in env_name:
                keys[env_name] = env_value
        
        # キーファイルからのキー
        try:
            file_keys = self._load_keys_file()
            keys.update(file_keys)
        except Exception as e:
            logger.warning(f"キーファイルからのキー一覧取得に失敗しました: {e}")
        
        return keys
    
    def generate_api_key(self, length: int = 32) -> str:
        """
        安全なAPIキーの生成
        
        Args:
            length: キーの長さ（バイト数）
        
        Returns:
            生成されたAPIキー
        """
        # ランダムな値を生成
        raw_key = secrets.token_bytes(length)
        
        # Base64エンコード
        api_key = base64.urlsafe_b64encode(raw_key).decode('utf-8')
        
        return api_key
    
    def _load_keys_file(self) -> Dict[str, str]:
        """
        暗号化されたキーファイルの読み込み
        
        Returns:
            キーの辞書
        """
        # キーファイルが存在しない場合は空の辞書を返す
        if not os.path.exists(self.keys_file):
            return {}
        
        # ファイルの読み込み
        with open(self.keys_file, 'r') as f:
            content = f.read().strip()
        
        # 空のファイルの場合は空の辞書を返す
        if not content:
            return {}
        
        # 暗号化されているかどうかを確認
        is_encrypted = content.startswith('ENCRYPTED:')
        
        if is_encrypted:
            # 暗号化されている場合は復号化
            if not CRYPTOGRAPHY_AVAILABLE:
                raise ValueError("暗号化されたキーファイルの復号化には、cryptographyライブラリが必要です")
            
            # 暗号化部分の抽出
            encrypted_data = content[len('ENCRYPTED:'):]
            
            # 復号化キーの取得
            if self.encryption_key is None:
                self._initialize_encryption_key()
            
            # 復号化
            try:
                fernet = Fernet(self.encryption_key)
                decrypted_data = fernet.decrypt(encrypted_data.encode('utf-8')).decode('utf-8')
                return json.loads(decrypted_data)
            except Exception as e:
                logger.error(f"キーファイルの復号化に失敗しました: {e}")
                return {}
        else:
            # 暗号化されていない場合はそのまま読み込み
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"キーファイルのJSONパースに失敗しました: {e}")
                return {}
    
    def _save_keys_file(self, keys: Dict[str, str], encrypt: bool = True):
        """
        キーファイルの保存
        
        Args:
            keys: 保存するキーの辞書
            encrypt: 暗号化するかどうか
        """
        # 暗号化する場合
        if encrypt and CRYPTOGRAPHY_AVAILABLE:
            # 暗号化キーの初期化
            if self.encryption_key is None:
                self._initialize_encryption_key()
            
            # JSONシリアライズ
            json_data = json.dumps(keys, indent=2)
            
            # 暗号化
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(json_data.encode('utf-8')).decode('utf-8')
            
            # ファイルに書き込み
            with open(self.keys_file, 'w') as f:
                f.write(f"ENCRYPTED:{encrypted_data}")
        else:
            # 暗号化なしでJSONとして保存
            with open(self.keys_file, 'w') as f:
                json.dump(keys, f, indent=2)
    
    def _initialize_encryption_key(self):
        """暗号化キーの初期化"""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ValueError("暗号化機能を使用するには、cryptographyライブラリが必要です")
        
        # マスターパスワードの取得
        password = self.master_password
        if password is None:
            password = getpass.getpass("マスターパスワードを入力してください: ")
        
        # ソルトファイルのパス
        salt_file = os.path.join(os.path.dirname(self.keys_file), '.salt')
        
        # ソルトの取得または生成
        if os.path.exists(salt_file):
            with open(salt_file, 'rb') as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, 'wb') as f:
                f.write(salt)
            os.chmod(salt_file, 0o600)  # 所有者のみ読み書き可能
        
        # キー導出関数
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        
        # 暗号化キーの生成
        key = base64.urlsafe_b64encode(kdf.derive(password.encode('utf-8')))
        self.encryption_key = key


class SecureTokenProvider:
    """
    セキュアなトークン生成プロバイダー
    
    一時的なトークン、認証コード、ワンタイムパスワードを生成するクラスです。
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        初期化
        
        Args:
            secret_key: 秘密鍵（指定されない場合は生成）
        """
        self.secret_key = secret_key or self._generate_secret_key()
    
    def generate_token(self, length: int = 32, expiration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """
        トークンの生成
        
        Args:
            length: トークンの長さ
            expiration_seconds: 有効期限（秒）
        
        Returns:
            トークン情報
        """
        import time
        
        # ランダムなトークンの生成
        token = secrets.token_urlsafe(length)
        
        # 有効期限の設定
        created_at = int(time.time())
        expires_at = None if expiration_seconds is None else created_at + expiration_seconds
        
        return {
            'token': token,
            'created_at': created_at,
            'expires_at': expires_at
        }
    
    def validate_token(self, token: str, expected_token: str, expires_at: Optional[int] = None) -> bool:
        """
        トークンの検証
        
        Args:
            token: 検証するトークン
            expected_token: 期待されるトークン
            expires_at: 有効期限のタイムスタンプ
        
        Returns:
            トークンが有効かどうか
        """
        import time
        
        # トークンの一致確認
        if not secrets.compare_digest(token, expected_token):
            return False
        
        # 有効期限の確認
        if expires_at is not None:
            current_time = int(time.time())
            if current_time > expires_at:
                return False
        
        return True
    
    def generate_otp(self, length: int = 6, expiration_seconds: int = 300) -> Dict[str, Any]:
        """
        ワンタイムパスワードの生成
        
        Args:
            length: パスワードの長さ
            expiration_seconds: 有効期限（秒）
        
        Returns:
            OTP情報
        """
        import time
        
        # 数字のみのOTPを生成
        otp = ''.join(secrets.choice('0123456789') for _ in range(length))
        
        # 有効期限の設定
        created_at = int(time.time())
        expires_at = created_at + expiration_seconds
        
        return {
            'otp': otp,
            'created_at': created_at,
            'expires_at': expires_at
        }
    
    def _generate_secret_key(self) -> str:
        """秘密鍵の生成"""
        return secrets.token_hex(32) 