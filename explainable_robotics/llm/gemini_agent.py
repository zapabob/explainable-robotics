"""
Gemini Proを使用したヒューマノイドロボットの頭脳エージェント

LangChainを使用してGemini Proをラップし、ヒューマノイドロボットの制御と学習を行います。
"""

import os
from typing import Dict, List, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

from ..utils.logging import get_logger

logger = get_logger(__name__)

class GeminiAgent:
    """Gemini Proを使用したヒューマノイドロボットの頭脳エージェント"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        初期化
        
        Args:
            api_key: Google AI Studio APIキー
            model_name: 使用するモデル名
            temperature: 生成時の温度パラメータ
            max_tokens: 最大トークン数
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google APIキーが必要です")
        
        # Gemini APIの設定
        genai.configure(api_key=self.api_key)
        
        # LangChainの設定
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=self.api_key
        )
        
        # 会話メモリの初期化
        self.memory = ConversationBufferMemory()
        
        # システムプロンプトの設定
        self.system_prompt = """あなたは高度なヒューマノイドロボットの頭脳です。
以下の能力を持っています：
1. 環境の認識と理解
2. 行動の計画と実行
3. 学習と適応
4. 人間との自然な対話

ロボットの制御と学習を行う際は、以下の点に注意してください：
- 安全性を最優先する
- 人間の意図を理解し、適切に応答する
- 新しい状況から学習し、行動を改善する
- 自身の状態を適切に管理する"""
        
        # 会話チェーンの初期化
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # 初期化メッセージの送信
        self.conversation.predict(input=self.system_prompt)
    
    def process_input(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        入力の処理と応答の生成
        
        Args:
            input_text: 入力テキスト
            context: 追加のコンテキスト情報
            
        Returns:
            生成された応答
        """
        # コンテキスト情報の追加
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            input_text = f"{context_str}\n\n{input_text}"
        
        # 応答の生成
        response = self.conversation.predict(input=input_text)
        return response
    
    def update_memory(self, new_memory: Dict[str, Any]):
        """
        メモリの更新
        
        Args:
            new_memory: 新しいメモリ情報
        """
        for key, value in new_memory.items():
            self.memory.save_context(
                {"input": f"Update {key}"},
                {"output": str(value)}
            )
    
    def get_memory(self) -> Dict[str, Any]:
        """
        現在のメモリの取得
        
        Returns:
            メモリの内容
        """
        return self.memory.load_memory_variables({})
    
    def clear_memory(self):
        """メモリのクリア"""
        self.memory.clear() 