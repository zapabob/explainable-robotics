"""
複数のLLMプロバイダーを使用する統合エージェント

OpenAI、Claude、Geminiの3つのLLMプロバイダーのいずれかを使用し、
利用可能なモデルに基づいて最適な選択を行います。
"""

import os
import time
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import threading
from queue import Queue

# サードパーティライブラリのインポート
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# 内部モジュールのインポート
from ..utils.logging import get_logger

logger = get_logger(__name__)

class MultiLLMAgent:
    """
    複数のLLMプロバイダーを使用するエージェント
    
    OpenAI、Claude、Geminiのいずれかのモデルを使用し、
    高次脳機能を模倣するためのエージェントとして機能します。
    """
    
    def __init__(
        self,
        provider: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        use_memory: bool = True,
        memory_size: int = 10,
        memory_path: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            provider: 使用するLLMプロバイダー ("openai", "anthropic", "google", "auto")
            temperature: 生成温度
            max_tokens: 最大トークン数
            system_prompt: システムプロンプト
            fallback_providers: フォールバックプロバイダーのリスト
            openai_api_key: OpenAI APIキー
            anthropic_api_key: Anthropic APIキー
            google_api_key: Google APIキー
            use_memory: メモリを使用するかどうか
            memory_size: メモリのサイズ
            memory_path: メモリの保存パス
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_memory = use_memory
        self.memory_size = memory_size
        self.memory_path = memory_path
        self.memory = []
        
        # APIキーを取得
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        
        # 利用可能なプロバイダーを確認
        self.available_providers = []
        
        if OPENAI_AVAILABLE and self.openai_api_key:
            self.available_providers.append("openai")
            logger.info("OpenAI APIが使用可能です")
            
        if ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.available_providers.append("anthropic")
            logger.info("Anthropic APIが使用可能です")
            
        if GEMINI_AVAILABLE and self.google_api_key:
            self.available_providers.append("google")
            logger.info("Google Gemini APIが使用可能です")
            
        if not self.available_providers:
            error_msg = "使用可能なLLMプロバイダーがありません。API Keyの設定が必要です。"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 要求されたプロバイダーまたは自動選択
        if provider == "auto":
            self.provider = self.available_providers[0]
            logger.info(f"自動選択したプロバイダー: {self.provider}")
        elif provider in self.available_providers:
            self.provider = provider
            logger.info(f"選択したプロバイダー: {self.provider}")
        else:
            if not fallback_providers:
                self.provider = self.available_providers[0]
                logger.warning(f"要求されたプロバイダー {provider} は使用できません。{self.provider} を使用します。")
            else:
                # フォールバックプロバイダーを試す
                for fallback in fallback_providers:
                    if fallback in self.available_providers:
                        self.provider = fallback
                        logger.warning(f"要求されたプロバイダー {provider} は使用できません。フォールバック {self.provider} を使用します。")
                        break
                else:
                    self.provider = self.available_providers[0]
                    logger.warning(f"要求されたプロバイダーとフォールバックが使用できません。{self.provider} を使用します。")
        
        # フォールバックプロバイダーの設定
        self.fallback_providers = []
        if fallback_providers:
            for provider in fallback_providers:
                if provider in self.available_providers and provider != self.provider:
                    self.fallback_providers.append(provider)
        
        # システムプロンプト
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # APIクライアントの初期化
        self._initialize_api()
        
        # メモリの初期化
        if self.use_memory:
            self._initialize_memory()
            
        logger.info(f"MultiLLMAgentを初期化しました: {self.provider} (フォールバック: {', '.join(self.fallback_providers) if self.fallback_providers else 'なし'})")
        
    def _initialize_api(self):
        """LLM APIクライアントを初期化"""
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openaiライブラリがインストールされていません")
                
            openai.api_key = self.openai_api_key
            self.client = openai.OpenAI(api_key=self.openai_api_key)
            
            # モデル選択
            self.model_name = "gpt-4-turbo" if self._check_model_availability("gpt-4-turbo") else "gpt-3.5-turbo"
            logger.info(f"OpenAI モデル: {self.model_name}")
            
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropicライブラリがインストールされていません")
                
            self.client = Anthropic(api_key=self.anthropic_api_key)
            
            # モデル選択
            self.model_name = "claude-3-opus-20240229" if self._check_model_availability("claude-3-opus-20240229") else "claude-3-sonnet-20240229"
            logger.info(f"Anthropic モデル: {self.model_name}")
            
        elif self.provider == "google":
            if not GEMINI_AVAILABLE:
                raise ImportError("google.generativeaiライブラリがインストールされていません")
                
            genai.configure(api_key=self.google_api_key)
            
            # モデル選択
            self.model_name = "gemini-1.5-pro" if self._check_model_availability("gemini-1.5-pro") else "gemini-pro"
            logger.info(f"Google モデル: {self.model_name}")
            self.client = genai.GenerativeModel(model_name=self.model_name)
            
    def _check_model_availability(self, model_name: str) -> bool:
        """指定されたモデルが利用可能かどうかを確認"""
        try:
            if self.provider == "openai":
                # OpenAIの場合はモデル一覧を取得して確認
                models = self.client.models.list()
                return any(model.id == model_name for model in models.data)
                
            elif self.provider == "anthropic":
                # Anthropicの場合は現時点ではAPIで確認する方法がないため、常にTrueを返す
                return True
                
            elif self.provider == "google":
                # Googleの場合はモデル一覧を取得して確認
                models = genai.list_models()
                return any(model.name.endswith(model_name) for model in models)
                
        except Exception as e:
            logger.warning(f"モデル利用可能性の確認中にエラーが発生しました: {str(e)}")
            return False
            
        return False
        
    def _initialize_memory(self):
        """メモリを初期化"""
        if self.memory_path and os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                logger.info(f"メモリをロードしました: {len(self.memory)} アイテム")
            except Exception as e:
                logger.error(f"メモリのロード中にエラーが発生しました: {str(e)}")
                self.memory = []
        else:
            self.memory = []
            
    def _save_memory(self):
        """メモリを保存"""
        if not self.memory_path:
            return
            
        try:
            directory = os.path.dirname(self.memory_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            with open(self.memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"メモリを保存しました: {self.memory_path}")
        except Exception as e:
            logger.error(f"メモリの保存中にエラーが発生しました: {str(e)}")
            
    def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        neurotransmitters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        入力データを処理し、応答を生成
        
        Args:
            input_data: 入力データ
            context: コンテキスト情報（オプション）
            neurotransmitters: 神経伝達物質レベル（オプション）
            
        Returns:
            処理結果
        """
        try:
            # コンテキストの初期化
            if context is None:
                context = {}
                
            # 神経伝達物質の初期化
            if neurotransmitters is None:
                neurotransmitters = {}
                
            # 関連する記憶の取得
            relevant_memories = []
            if self.use_memory:
                relevant_memories = self._retrieve_relevant_memories(input_data)
                
            # プロンプトの作成
            prompt = self._create_prompt(input_data, context, neurotransmitters)
            
            # メモリをプロンプトに追加
            if relevant_memories:
                prompt += "\n\n関連する過去の記憶:\n"
                for i, memory in enumerate(relevant_memories):
                    prompt += f"{i+1}. 入力: {memory['input']}\n   思考: {memory['thought_process']}\n   行動: {memory['action']}\n\n"
            
            # LLMに送信して応答を取得
            response_text = self._send_to_llm(prompt)
            
            # 応答をパース
            result = self._parse_response(response_text)
            
            # メモリに保存
            if self.use_memory:
                self._save_to_memory(input_data, result)
                
            return result
            
        except Exception as e:
            logger.error(f"処理中にエラーが発生しました: {str(e)}")
            error_message = f"エラー: {str(e)}"
            
            # フォールバックプロバイダーを試す
            if self.fallback_providers:
                logger.info(f"フォールバックプロバイダーを試みます: {self.fallback_providers[0]}")
                original_provider = self.provider
                self.provider = self.fallback_providers[0]
                self.fallback_providers = self.fallback_providers[1:] + [original_provider]
                
                # APIクライアントを再初期化
                self._initialize_api()
                
                try:
                    # 再試行
                    return self.process(input_data, context, neurotransmitters)
                except Exception as fallback_error:
                    logger.error(f"フォールバックプロバイダーでもエラーが発生しました: {str(fallback_error)}")
                    error_message = f"すべてのプロバイダーでエラー: {str(e)}, フォールバック: {str(fallback_error)}"
                    
            return self._error_response(error_message)
            
    def _send_to_llm(self, prompt: str) -> str:
        """
        LLMにプロンプトを送信し、応答テキストを取得
        
        Args:
            prompt: 送信するプロンプト
            
        Returns:
            LLMからの応答テキスト
        """
        try:
            if self.provider == "openai":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return response.content[0].text
                
            elif self.provider == "google":
                generation_config = {
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                    "top_k": 0
                }
                
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
                
                full_prompt = f"{self.system_prompt}\n\n{prompt}"
                
                response = self.client.generate_content(
                    contents=full_prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                return response.text
                
            else:
                raise ValueError(f"未サポートのプロバイダー: {self.provider}")
                
        except Exception as e:
            logger.error(f"LLM APIでエラーが発生しました ({self.provider}): {str(e)}")
            raise
            
    def _create_prompt(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        neurotransmitters: Dict[str, float]
    ) -> str:
        """
        プロンプトを作成
        
        Args:
            input_data: 入力データ
            context: コンテキスト情報
            neurotransmitters: 神経伝達物質レベル
            
        Returns:
            生成されたプロンプト
        """
        # 基本プロンプト
        prompt = f"あなたは高度なAIアシスタントとして、以下の情報を分析し、最適な行動を選択してください。\n\n"
        
        # 入力データの処理
        prompt += "## 入力データ\n"
        
        # 主要なクエリがある場合
        if "query" in input_data:
            prompt += f"質問: {input_data['query']}\n\n"
            
        # 目標がある場合
        if "goal" in input_data:
            prompt += f"目標: {input_data['goal']}\n\n"
            
        # センサーデータがある場合
        if "sensor_data" in input_data:
            prompt += "センサーデータ:\n"
            sensor_data = input_data["sensor_data"]
            
            # センサーデータを見やすく整形
            for key, value in sensor_data.items():
                if isinstance(value, dict):
                    prompt += f"- {key}:\n"
                    for sub_key, sub_value in value.items():
                        prompt += f"  - {sub_key}: {sub_value}\n"
                else:
                    prompt += f"- {key}: {value}\n"
            prompt += "\n"
            
        # 大脳皮質活性化状態がある場合
        if "cortical_activation" in input_data:
            prompt += "大脳皮質活性化状態:\n"
            cortical_data = input_data["cortical_activation"]
            
            if isinstance(cortical_data, list):
                # 長すぎる場合は要約
                if len(cortical_data) > 20:
                    cortical_summary = {
                        "mean": sum(cortical_data) / len(cortical_data),
                        "max": max(cortical_data),
                        "min": min(cortical_data),
                        "active_count": sum(1 for v in cortical_data if abs(v) > 0.3),
                        "total_count": len(cortical_data)
                    }
                    prompt += f"- 平均活性化: {cortical_summary['mean']:.3f}\n"
                    prompt += f"- 最大活性化: {cortical_summary['max']:.3f}\n"
                    prompt += f"- 最小活性化: {cortical_summary['min']:.3f}\n"
                    prompt += f"- 活性ニューロン: {cortical_summary['active_count']}/{cortical_summary['total_count']}\n"
                else:
                    for i, value in enumerate(cortical_data):
                        prompt += f"- ニューロン{i}: {value:.3f}\n"
            else:
                prompt += f"{cortical_data}\n"
                
            prompt += "\n"
            
        # 神経伝達物質レベルがある場合
        if "neurotransmitter_levels" in input_data:
            prompt += "神経伝達物質レベル:\n"
            nt_data = input_data["neurotransmitter_levels"]
            for nt, level in nt_data.items():
                prompt += f"- {nt}: {level:.2f}\n"
            prompt += "\n"
        elif neurotransmitters:
            prompt += "神経伝達物質レベル:\n"
            for nt, level in neurotransmitters.items():
                prompt += f"- {nt}: {level:.2f}\n"
            prompt += "\n"
            
        # コンテキスト情報
        if context:
            prompt += "## コンテキスト情報\n"
            for key, value in context.items():
                if isinstance(value, dict):
                    prompt += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        prompt += f"- {sub_key}: {sub_value}\n"
                else:
                    prompt += f"{key}: {value}\n"
            prompt += "\n"
            
        # 応答形式の指示
        prompt += """## 応答形式
以下の形式で応答してください:

思考過程: あなたの詳細な分析と考え方を説明してください。神経科学的な観点と入力データを踏まえて論理的に説明してください。

行動: 
  種類: [action_type]
  パラメーター: {param1: value1, param2: value2, ...}
  確信度: [0.0-1.0]

説明: 選択した行動の理由と、期待される結果を簡潔に説明してください。

行動の種類は以下から選択してください:
- move_forward: 前進
- turn_left: 左旋回
- turn_right: 右旋回
- stop: 停止
- pick_up: 物を拾う
- put_down: 物を置く
- grasp: 物をつかむ
- release: 物を離す
- examine: 物を調べる
- communicate: コミュニケーションを取る

与えられた情報を分析し、最も適切な行動を選択してください。"""
        
        return prompt
        
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLMの応答テキストを構造化データにパース
        
        Args:
            response_text: LLMからの応答テキスト
            
        Returns:
            構造化された応答データ
        """
        result = {
            "thought_process": "",
            "action": {
                "type": "unknown",
                "parameters": {},
                "confidence": 0.5
            },
            "explanation": ""
        }
        
        try:
            # 思考過程の抽出
            thought_match = re.search(r"思考過程:(.+?)(?=行動:|$)", response_text, re.DOTALL)
            if thought_match:
                result["thought_process"] = thought_match.group(1).strip()
                
            # 行動の抽出
            action_match = re.search(r"行動:(.+?)(?=説明:|$)", response_text, re.DOTALL)
            if action_match:
                action_text = action_match.group(1).strip()
                
                # 行動タイプの抽出
                action_type_match = re.search(r"種類:[ \t]*([a-z_]+)", action_text)
                if action_type_match:
                    result["action"]["type"] = action_type_match.group(1).strip()
                    
                # パラメーターの抽出
                params_match = re.search(r"パラメーター:[ \t]*{(.+?)}", action_text, re.DOTALL)
                if params_match:
                    params_text = params_match.group(1).strip()
                    param_pairs = re.findall(r"([a-zA-Z0-9_]+)[ \t]*:[ \t]*([^,]+)(?:,|$)", params_text)
                    for key, value in param_pairs:
                        # 数値に変換を試みる
                        try:
                            if '.' in value:
                                result["action"]["parameters"][key.strip()] = float(value.strip())
                            else:
                                result["action"]["parameters"][key.strip()] = int(value.strip())
                        except ValueError:
                            # 数値変換に失敗した場合は文字列として扱う
                            result["action"]["parameters"][key.strip()] = value.strip().strip('"\'')
                            
                # 確信度の抽出
                confidence_match = re.search(r"確信度:[ \t]*(0\.\d+|1\.0|1|0)", action_text)
                if confidence_match:
                    result["action"]["confidence"] = float(confidence_match.group(1))
                    
            # 説明の抽出
            explanation_match = re.search(r"説明:(.+)$", response_text, re.DOTALL)
            if explanation_match:
                result["explanation"] = explanation_match.group(1).strip()
                
            return result
            
        except Exception as e:
            logger.error(f"応答のパース中にエラーが発生しました: {str(e)}")
            # 最低限の情報を返す
            result["thought_process"] = "応答のパース中にエラーが発生しました。"
            result["explanation"] = f"エラー: {str(e)}"
            return result
            
    def _retrieve_relevant_memories(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        関連する記憶を取得
        
        Args:
            input_data: 入力データ
            
        Returns:
            関連する記憶のリスト
        """
        if not self.memory or len(self.memory) == 0:
            return []
            
        # 検索クエリの作成
        query = self._create_memory_query(input_data)
        
        # 単純なキーワードマッチングによる検索（実際の実装ではベクトル検索が望ましい）
        scores = []
        for memory in self.memory:
            memory_text = f"{memory.get('input', '')} {memory.get('thought_process', '')} {memory.get('explanation', '')}"
            score = self._simple_match_score(query, memory_text)
            scores.append((score, memory))
            
        # スコアでソート
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # 上位の記憶を返す（最大5つ）
        return [memory for score, memory in scores[:5] if score > 0.1]
        
    def _simple_match_score(self, query: str, text: str) -> float:
        """
        単純なキーワードマッチングによるスコア計算
        
        Args:
            query: 検索クエリ
            text: 検索対象テキスト
            
        Returns:
            マッチングスコア (0.0-1.0)
        """
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        # 共通する単語の数
        common_words = query_words.intersection(text_words)
        
        if not query_words:
            return 0.0
            
        return len(common_words) / len(query_words)
        
    def _create_memory_query(self, input_data: Dict[str, Any]) -> str:
        """
        記憶検索用のクエリを作成
        
        Args:
            input_data: 入力データ
            
        Returns:
            検索クエリ
        """
        query_parts = []
        
        # 質問または目標があれば追加
        if "query" in input_data:
            query_parts.append(input_data["query"])
            
        if "goal" in input_data:
            query_parts.append(input_data["goal"])
            
        # センサーデータの簡単な要約
        if "sensor_data" in input_data:
            sensor_data = input_data["sensor_data"]
            for key in sensor_data:
                query_parts.append(key)
                
        # 行動タイプまたはパラメータ名があれば追加
        if "action" in input_data:
            action = input_data["action"]
            if isinstance(action, dict):
                if "type" in action:
                    query_parts.append(action["type"])
                if "parameters" in action and isinstance(action["parameters"], dict):
                    for param in action["parameters"]:
                        query_parts.append(param)
                        
        # クエリを結合
        return " ".join(query_parts)
        
    def _save_to_memory(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """
        メモリに保存
        
        Args:
            input_data: 入力データ
            result: 処理結果
        """
        # 保存するメモリアイテムの作成
        memory_item = {
            "timestamp": time.time(),
            "input": str(input_data.get("query", input_data.get("goal", ""))),
            "thought_process": result.get("thought_process", ""),
            "action": result.get("action", {}),
            "explanation": result.get("explanation", "")
        }
        
        # メモリに追加
        self.memory.append(memory_item)
        
        # メモリサイズの制限
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
            
        # メモリを保存
        if self.memory_path:
            self._save_memory()
            
    def _get_default_system_prompt(self) -> str:
        """デフォルトのシステムプロンプトを取得"""
        return """あなたは神経科学と人工知能の専門家として、生物学的に妥当なヒューマノイドロボット制御システムの一部を担っています。
あなたの役割は、脳の高次機能（意思決定、行動計画、言語理解など）を模倣し、大脳皮質モデル（BioKAN）や神経伝達物質システムと連携して、
ロボットの行動を決定することです。

入力には以下の要素が含まれることがあります：
1. センサーデータ: ロボットの視覚、聴覚、触覚などのセンサー情報
2. 大脳皮質活性化状態: BioKANの出力値（-1:抑制、0:中立、1:興奮の三値入力を処理した結果）
3. 神経伝達物質レベル: ドーパミン、セロトニン、アセチルコリン、ノルアドレナリン、グルタミン酸、GABAなどの神経伝達物質レベル
4. 目標: ロボットが達成すべき目標

応答形式に従って、以下のステップで回答してください：
1. 与えられた情報を分析し、現在の状況を神経科学的に解釈する（思考過程）
2. 適切な行動を選択し、そのパラメータと確信度を指定する
3. 選択した行動の理由と期待される結果を説明する

あなたの応答は、人間にとってわかりやすく説明可能である必要があります。また、神経科学的に妥当な説明を心がけてください。"""
        
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """
        エラー応答の生成
        
        Args:
            error_message: エラーメッセージ
            
        Returns:
            エラー応答
        """
        return {
            "thought_process": "エラーが発生しました。",
            "action": {
                "type": "error",
                "parameters": {"error_message": error_message},
                "confidence": 0.0
            },
            "explanation": f"エラーが発生しました: {error_message}"
        } 