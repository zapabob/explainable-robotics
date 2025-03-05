"""
ロギング機能のユーティリティモジュール
"""

import os
import sys
import logging
import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

# ロガーのデフォルト設定
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_COLOR_FORMAT = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "explainable_robotics.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 5

# カラー出力用の定数
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

# ログレベルごとの色
LEVEL_COLORS = {
    logging.DEBUG: COLORS['BLUE'],
    logging.INFO: COLORS['GREEN'],
    logging.WARNING: COLORS['YELLOW'],
    logging.ERROR: COLORS['RED'],
    logging.CRITICAL: COLORS['BOLD'] + COLORS['RED']
}

class ColoredFormatter(logging.Formatter):
    """カラー出力対応のフォーマッタ"""
    
    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        """
        初期化
        
        Args:
            fmt: フォーマット文字列
            datefmt: 日付フォーマット
            style: フォーマットスタイル
            use_color: カラー出力を使用するかどうか
        """
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color
    
    def format(self, record):
        """
        レコードのフォーマット
        
        Args:
            record: ログレコード
        
        Returns:
            フォーマットされたログメッセージ
        """
        # オリジナルのメッセージをフォーマット
        message = super().format(record)
        
        # カラー出力が無効または対応するレベルの色がない場合はそのまま返す
        if not self.use_color or record.levelno not in LEVEL_COLORS:
            return message
        
        # レベルに対応する色を適用
        color = LEVEL_COLORS[record.levelno]
        reset = COLORS['RESET']
        
        # 色をメッセージに適用（Windowsの場合は色をスキップ）
        if sys.platform.startswith('win') and not os.environ.get('FORCE_COLOR'):
            return message
            
        return f"{color}{message}{reset}"


def get_logger(name=None, log_level=None, log_file=None, console_output=True, 
               color_output=True, max_file_size_mb=None, backup_count=None):
    """
    指定された名前でロガーを取得または作成する

    Args:
        name (str, optional): ロガーの名前。デフォルトはNone（ルートロガー）
        log_level (int, optional): ロギングレベル。デフォルトはINFO
        log_file (str, optional): ログファイルのパス。デフォルトは'logs/explainable_robotics.log'
        console_output (bool, optional): コンソール出力を有効にするかどうか。デフォルトはTrue
        color_output (bool, optional): カラー出力を有効にするかどうか。デフォルトはTrue
        max_file_size_mb (int, optional): ログファイルの最大サイズ（MB単位）。デフォルトは10MB
        backup_count (int, optional): 保持するバックアップファイルの数。デフォルトは5

    Returns:
        logging.Logger: 設定されたロガーオブジェクト
    """
    # デフォルト値の設定
    if log_level is None:
        log_level = DEFAULT_LOG_LEVEL
    if log_file is None:
        log_file = os.path.join(DEFAULT_LOG_DIR, DEFAULT_LOG_FILE)
    if max_file_size_mb is None:
        max_file_size_mb = DEFAULT_MAX_BYTES // (1024 * 1024)
    if backup_count is None:
        backup_count = DEFAULT_BACKUP_COUNT

    # ロガーの取得
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # すでにハンドラが設定されている場合は何もしない
    if logger.handlers:
        return logger

    # ファイルハンドラの設定
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_file_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # コンソール出力が有効な場合
    if console_output:
        console_handler = logging.StreamHandler()
        
        # カラー出力が有効な場合
        if color_output:
            color_formatter = ColoredFormatter(
                DEFAULT_COLOR_FORMAT,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(color_formatter)
        else:
            console_handler.setFormatter(file_formatter)
            
        logger.addHandler(console_handler)

    return logger 


class LogContext:
    """
    ログコンテキスト管理
    
    with文でログ出力の追加コンテキストを管理します。
    """
    
    def __init__(self, logger, context=None):
        """
        初期化
        
        Args:
            logger: ロガーインスタンス
            context: 追加コンテキスト情報
        """
        self.logger = logger
        self.context = context or {}
        self.old_context = {}
        
    def __enter__(self):
        """コンテキスト開始"""
        # 現在のコンテキストを保存
        for handler in self.logger.handlers:
            if hasattr(handler, 'formatter') and hasattr(handler.formatter, '_style'):
                if not hasattr(handler.formatter._style, '_context'):
                    handler.formatter._style._context = {}
                
                self.old_context[handler] = handler.formatter._style._context.copy()
                
                # 新しいコンテキストを追加
                handler.formatter._style._context.update(self.context)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了"""
        # 元のコンテキストに戻す
        for handler in self.logger.handlers:
            if handler in self.old_context and hasattr(handler, 'formatter') and hasattr(handler.formatter, '_style'):
                handler.formatter._style._context = self.old_context[handler]


def setup_global_logging(log_level='INFO', log_dir='logs'):
    """
    グローバルロギング設定
    
    Args:
        log_level: ログレベル
        log_dir: ログディレクトリ
    """
    # ログディレクトリの作成
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 現在の日時を取得してログファイル名を作成
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'explainable_robotics_{timestamp}.log')
    
    # ルートロガーの設定
    root_logger = get_logger(
        None, 
        log_level=log_level, 
        log_file=log_file, 
        console_output=True,
        color_output=True,
        max_file_size_mb=10,
        backup_count=5
    )
    
    # ライブラリのロギング制御
    # 一部のライブラリのログレベルを調整
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    return root_logger 