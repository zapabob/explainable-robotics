"""Core Module

ロボットの認知機能と行動決定のためのコアモジュール。
Gemini APIを利用した知的エージェントを提供します。
"""

import logging
import sys

logger = logging.getLogger(__name__)

# 安全にモジュールをインポートするためのトライキャッチブロック
try:
    from .gemini_agent import GeminiAgent
    GEMINI_AVAILABLE = True
except (ImportError, TypeError) as e:
    GEMINI_AVAILABLE = False
    logger.warning(f"GeminiAgentをインポートできません: {e}")
    # フォールバックのダミークラスを提供
    class GeminiAgent:
        """GeminiAgentのダミー実装"""
        def __init__(self, *args, **kwargs):
            logger.warning("GeminiAgentはインポートできないため、ダミー実装を使用しています")
        
        def process(self, *args, **kwargs):
            """ダミー処理メソッド"""
            return {"error": "GeminiAgentは利用できません"}

# エラーによる依存関係の問題が発生した場合に備えて、他のインポートも安全に行う
try:
    from .multi_llm_agent import MultiLLMAgent
    MULTI_LLM_AVAILABLE = True
except (ImportError, TypeError) as e:
    MULTI_LLM_AVAILABLE = False
    logger.warning(f"MultiLLMAgentをインポートできません: {e}")
    # ダミークラス
    class MultiLLMAgent:
        """MultiLLMAgentのダミー実装"""
        def __init__(self, *args, **kwargs):
            logger.warning("MultiLLMAgentはインポートできないため、ダミー実装を使用しています")
        
        def process(self, *args, **kwargs):
            """ダミー処理メソッド"""
            return {"error": "MultiLLMAgentは利用できません"}

# 必要に応じて他のモジュールもここに追加

__all__ = ["GeminiAgent", "MultiLLMAgent"]
