"""Gemini Agent

ヒューマノイドロボットの制御のためのGemini APIベースのエージェント。
ロボットの行動決定、言語理解、環境解析を担当します。
"""

import os
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# APIアクセスのためのライブラリ
try:
    import google.generativeai as genai
    from google.generativeai import GenerativeModel
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("WARNING: Google Generative AIライブラリが利用できません。モックモードで実行します。")

# ベクトル埋め込みとメモリのためのライブラリ
try:
    import langchain
    from langchain.embeddings import VertexAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.memory import ConversationBufferMemory
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("WARNING: Langchainライブラリが利用できません。メモリと埋め込み機能は無効化されます。")

# ローカルの依存関係
from ..utils.logging import get_logger
from ..utils.security import SecureKeyManager

# ロガーの設定
logger = get_logger(__name__)


class GeminiAgent:
    """
    Gemini APIを利用した知的エージェント
    
    ロボットの意思決定、環境理解、および行動計画を担当します。
    BioKANモデルと連携して、生物学的に妥当な意思決定を行います。
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        safety_threshold: str = "BLOCK_MEDIUM_AND_ABOVE",
        use_memory: bool = True,
        memory_size: int = 10,
        memory_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        初期化
        
        Args:
            api_key: Gemini API キー (Noneの場合は環境変数から取得)
            model_name: 使用するモデル名
            temperature: 温度パラメータ (0.0-1.0)
            max_output_tokens: 最大出力トークン数
            safety_threshold: 安全性しきい値
            use_memory: メモリを使用するかどうか
            memory_size: 会話履歴の保持数
            memory_path: メモリの保存パス
            system_prompt: システムプロンプト
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.safety_threshold = safety_threshold
        self.use_memory = use_memory and LANGCHAIN_AVAILABLE
        self.memory_size = memory_size
        self.memory_path = memory_path
        
        # デフォルトのシステムプロンプト
        self.default_system_prompt = """
        あなたはヒューマノイドロボットの制御システムの中核を担う高度なAIエージェントです。
        入力された環境データと内部状態に基づいて、最適な行動を決定してください。
        
        あなたの主な役割は以下の通りです：
        1. 環境の理解: センサーデータを解析し、周囲の状況を理解します
        2. 行動計画: 目標達成のための最適な行動を計画します
        3. 感情理解: 人間の感情や意図を理解し、適切に応答します
        4. 安全性確保: すべての行動において安全性を最優先します
        5. 説明可能性: 決定の理由を明確に説明できるようにします
        
        出力形式は以下のJSON形式に従ってください:
        {
            "thought_process": "内部思考過程の説明",
            "action": {
                "type": "行動タイプ",
                "parameters": {},
                "confidence": 0.0-1.0
            },
            "explanation": "行動選択の理由の説明"
        }
        """
        
        # システムプロンプトの設定
        self.system_prompt = system_prompt if system_prompt else self.default_system_prompt
        
        # セッションID
        self.session_id = str(uuid.uuid4())
        
        # API初期化
        self._initialize_api(api_key)
        
        # メモリの初期化
        self._initialize_memory()
        
        logger.info(f"GeminiAgentを初期化しました（モデル: {model_name}）")
    
    def _initialize_api(self, api_key: Optional[str] = None):
        """
        API初期化
        
        Args:
            api_key: API キー
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini APIが利用できないため、モックモードで動作します")
            self.gemini = None
            return
        
        try:
            # APIキーの取得
            if api_key is None:
                key_manager = SecureKeyManager()
                api_key = key_manager.get_key("GOOGLE_API_KEY")
            
            if not api_key:
                api_key = os.environ.get("GOOGLE_API_KEY")
            
            if not api_key:
                raise ValueError("API キーが指定されていません。引数または環境変数で指定してください。")
            
            # Gemini API設定
            genai.configure(api_key=api_key)
            
            # 生成モデルのセットアップ
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "top_p": 0.95,
                "top_k": 64
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": self.safety_threshold
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": self.safety_threshold
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": self.safety_threshold
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": self.safety_threshold
                }
            ]
            
            self.gemini = GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            logger.info("Gemini APIの初期化に成功しました")
            
        except Exception as e:
            logger.error(f"Gemini APIの初期化に失敗しました: {e}")
            self.gemini = None
    
    def _initialize_memory(self):
        """メモリの初期化"""
        self.memory = None
        self.vector_store = None
        
        if not self.use_memory:
            return
        
        if not LANGCHAIN_AVAILABLE:
            logger.warning("Langchainが利用できないため、メモリ機能は無効化されます")
            return
        
        try:
            # 会話メモリの初期化
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.memory_size * 1000  # 概算
            )
            
            # ベクトルストアの初期化
            if self.memory_path:
                os.makedirs(self.memory_path, exist_ok=True)
                
                # ベクトル埋め込みの初期化
                embeddings = VertexAIEmbeddings()
                
                # ベクトルストアの初期化
                self.vector_store = Chroma(
                    collection_name="robot_memory",
                    embedding_function=embeddings,
                    persist_directory=self.memory_path
                )
                
                logger.info(f"メモリを初期化しました（保存先: {self.memory_path}）")
            
        except Exception as e:
            logger.error(f"メモリの初期化に失敗しました: {e}")
            self.memory = None
            self.vector_store = None
    
    def process(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        neurotransmitters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        入力データの処理と行動決定
        
        Args:
            input_data: 入力データ（センサー情報、環境状態など）
            context: 追加コンテキスト情報
            neurotransmitters: 神経伝達物質レベル（BioKANモデルから）
        
        Returns:
            処理結果と行動計画
        """
        # コンテキストの準備
        if context is None:
            context = {}
        
        # 神経伝達物質の準備
        if neurotransmitters is None:
            neurotransmitters = {}
        
        # プロンプトの作成
        prompt = self._create_prompt(input_data, context, neurotransmitters)
        
        # メモリからの関連情報取得
        relevant_memories = self._retrieve_relevant_memories(input_data)
        
        # モックモードチェック
        if not GEMINI_AVAILABLE or self.gemini is None:
            return self._mock_response(input_data)
        
        try:
            # Gemini APIでの生成
            response = self.gemini.generate_content(prompt)
            
            # 応答の解析
            result = self._parse_response(response.text)
            
            # メモリへの保存
            if self.use_memory:
                self._save_to_memory(input_data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini APIでのレスポンス生成に失敗しました: {e}")
            return self._error_response(str(e))
    
    def _create_prompt(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        neurotransmitters: Dict[str, float]
    ) -> str:
        """
        プロンプトの作成
        
        Args:
            input_data: 入力データ
            context: コンテキスト情報
            neurotransmitters: 神経伝達物質レベル
        
        Returns:
            生成されたプロンプト
        """
        # システムプロンプト
        prompt = f"{self.system_prompt}\n\n"
        
        # 入力データの追加
        prompt += "## 入力データ\n"
        prompt += f"```json\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n```\n\n"
        
        # コンテキストの追加
        prompt += "## コンテキスト\n"
        prompt += f"```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```\n\n"
        
        # 神経伝達物質レベルの追加
        prompt += "## 神経伝達物質レベル\n"
        prompt += f"```json\n{json.dumps(neurotransmitters, ensure_ascii=False, indent=2)}\n```\n\n"
        
        # レスポンス指示
        prompt += "## 指示\n"
        prompt += "上記の情報を分析し、最適な行動を決定してください。応答は指定されたJSON形式で出力してください。\n"
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        レスポンスの解析
        
        Args:
            response_text: レスポンステキスト
        
        Returns:
            解析された応答データ
        """
        try:
            # JSONの抽出（マークダウンコードブロックからの抽出対応）
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_text = response_text.strip()
            
            # JSONのパース
            data = json.loads(json_text)
            
            # 必須フィールドの検証
            required_fields = ["thought_process", "action", "explanation"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"必須フィールド '{field}' がレスポンスに含まれていません")
            
            # 行動データの検証
            action = data["action"]
            if not isinstance(action, dict):
                raise ValueError("'action' フィールドは辞書である必要があります")
            
            if "type" not in action:
                raise ValueError("'action.type' フィールドが必要です")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSONのパースに失敗しました: {e}")
            logger.debug(f"解析できなかったテキスト: {response_text}")
            return self._error_response(f"JSONのパースに失敗しました: {e}")
            
        except Exception as e:
            logger.error(f"レスポンスの解析に失敗しました: {e}")
            return self._error_response(f"レスポンスの解析に失敗しました: {e}")
    
    def _retrieve_relevant_memories(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        関連するメモリの取得
        
        Args:
            input_data: 入力データ
        
        Returns:
            関連するメモリのリスト
        """
        if not self.use_memory or not self.vector_store:
            return []
        
        try:
            # クエリの作成
            query = self._create_memory_query(input_data)
            
            # ベクトル検索
            results = self.vector_store.similarity_search(
                query=query,
                k=3  # 上位3件を取得
            )
            
            # 結果の変換
            memories = []
            for doc in results:
                try:
                    memory = json.loads(doc.page_content)
                    memories.append(memory)
                except json.JSONDecodeError:
                    # テキスト形式のメモリの場合
                    memories.append({"text": doc.page_content, "metadata": doc.metadata})
            
            return memories
            
        except Exception as e:
            logger.error(f"メモリの取得に失敗しました: {e}")
            return []
    
    def _create_memory_query(self, input_data: Dict[str, Any]) -> str:
        """
        メモリクエリの作成
        
        Args:
            input_data: 入力データ
        
        Returns:
            クエリ文字列
        """
        # 環境データの抽出
        environment = input_data.get("environment", {})
        
        # 位置情報の抽出
        location = environment.get("location", {})
        location_str = f"{location.get('x', 0)}, {location.get('y', 0)}, {location.get('z', 0)}"
        
        # オブジェクト情報の抽出
        objects = environment.get("detected_objects", [])
        object_names = [obj.get("name", "unknown") for obj in objects[:3]]
        object_str = ", ".join(object_names)
        
        # クエリ文字列の作成
        query = f"位置: {location_str}, オブジェクト: {object_str}"
        
        # 目標情報の追加
        goal = input_data.get("goal", "")
        if goal:
            query += f", 目標: {goal}"
        
        return query
    
    def _save_to_memory(self, input_data: Dict[str, Any], result: Dict[str, Any]):
        """
        メモリへの保存
        
        Args:
            input_data: 入力データ
            result: 処理結果
        """
        if not self.use_memory:
            return
        
        try:
            # 会話メモリへの保存
            if self.memory:
                self.memory.save_context(
                    {"input": json.dumps(input_data, ensure_ascii=False)},
                    {"output": json.dumps(result, ensure_ascii=False)}
                )
            
            # ベクトルストアへの保存
            if self.vector_store:
                # メモリエントリの作成
                memory_entry = {
                    "timestamp": time.time(),
                    "session_id": self.session_id,
                    "input": input_data,
                    "result": result
                }
                
                # ドキュメントの作成
                doc = Document(
                    page_content=json.dumps(memory_entry, ensure_ascii=False),
                    metadata={
                        "timestamp": memory_entry["timestamp"],
                        "session_id": memory_entry["session_id"],
                        "action_type": result.get("action", {}).get("type", "unknown")
                    }
                )
                
                # ベクトルストアへの追加
                self.vector_store.add_documents([doc])
                
                # 永続化（設定されている場合）
                if hasattr(self.vector_store, "persist"):
                    self.vector_store.persist()
            
        except Exception as e:
            logger.error(f"メモリへの保存に失敗しました: {e}")
    
    def _mock_response(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        モックレスポンスの生成
        
        Args:
            input_data: 入力データ
        
        Returns:
            モックレスポンス
        """
        logger.warning("モックモードでレスポンスを生成します")
        
        # 基本的な行動タイプ
        action_types = ["move_forward", "turn_left", "turn_right", "stop", "greet", "pick_up"]
        
        # 入力に基づく簡単な行動選択
        action_type = "unknown"
        
        goal = input_data.get("goal", "")
        if "移動" in goal or "歩く" in goal:
            action_type = "move_forward"
        elif "挨拶" in goal or "greeting" in goal:
            action_type = "greet"
        elif "停止" in goal or "止まる" in goal:
            action_type = "stop"
        else:
            # ランダム選択
            import random
            action_type = random.choice(action_types)
        
        return {
            "thought_process": "これはモックレスポンスです。実際のGemini APIレスポンスではありません。",
            "action": {
                "type": action_type,
                "parameters": {},
                "confidence": 0.7
            },
            "explanation": "モックモードでの自動選択です。"
        }
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """
        エラーレスポンスの生成
        
        Args:
            error_message: エラーメッセージ
        
        Returns:
            エラーレスポンス
        """
        return {
            "thought_process": "エラーが発生しました。",
            "action": {
                "type": "error",
                "parameters": {
                    "error_message": error_message
                },
                "confidence": 0.0
            },
            "explanation": f"エラーが発生しました: {error_message}"
        } 