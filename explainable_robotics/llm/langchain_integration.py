import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import base64
from pathlib import Path

import numpy as np
import torch

# LangChain関連のインポート
from langchain.llms import OpenAI, GooglePalm
from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatGooglePalm
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import GooglePalmEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader

# ローカルLLM用
try:
    from langchain.llms import CTransformers, LlamaCpp
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultimodalLLMSystem:
    """
    複数のLLMプロバイダーとマルチモーダル機能を統合するシステム。
    ロボット制御のための高度な推論、説明生成、知識検索を提供します。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # デフォルト設定
        self.config = {
            'providers': {
                'openai': {
                    'enabled': False,
                    'api_key': None,
                    'model': 'gpt-4-vision-preview'
                },
                'google': {
                    'enabled': False,
                    'api_key': None,
                    'model': 'gemini-1.0-pro-vision'
                },
                'anthropic': {
                    'enabled': False,
                    'api_key': None,
                    'model': 'claude-3-opus-20240229'
                },
                'local': {
                    'enabled': False,
                    'model_path': None,
                    'model_type': 'llama',  # llama, mistral, phi など
                    'context_length': 2048
                }
            },
            'embeddings': {
                'provider': 'openai',  # openai, google, huggingface, local
                'model': 'text-embedding-ada-002'
            },
            'memory': {
                'max_tokens': 4000,
                'persist_directory': './memory'
            },
            'default_provider': 'openai'
        }
        
        # 設定ファイルがある場合は読み込む
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 既存の設定にマージ
                    self._merge_config(loaded_config)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗しました: {str(e)}")
        
        # 環境変数からAPIキーを取得（設定よりも優先）
        self._load_api_keys_from_env()
        
        # LLMモデルの初期化
        self.llms = {}
        self.chat_models = {}
        self.embedding_model = None
        self._initialize_models()
        
        # 会話メモリの初期化
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=self.config['memory']['max_tokens']
        )
        
        # 知識ベースの初期化
        self.knowledge_base = None
        
        logger.info("マルチモーダルLLMシステムを初期化しました")
        
    def _merge_config(self, new_config: Dict) -> None:
        """設定を再帰的にマージします"""
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self._merge_config(value)
                else:
                    self.config[key] = value
                    
    def _load_api_keys_from_env(self) -> None:
        """環境変数からAPIキーを読み込みます"""
        # OpenAI
        if 'OPENAI_API_KEY' in os.environ:
            self.config['providers']['openai']['api_key'] = os.environ['OPENAI_API_KEY']
            self.config['providers']['openai']['enabled'] = True
            
        # Google
        if 'GOOGLE_API_KEY' in os.environ:
            self.config['providers']['google']['api_key'] = os.environ['GOOGLE_API_KEY']
            self.config['providers']['google']['enabled'] = True
            
        # Anthropic
        if 'ANTHROPIC_API_KEY' in os.environ:
            self.config['providers']['anthropic']['api_key'] = os.environ['ANTHROPIC_API_KEY']
            self.config['providers']['anthropic']['enabled'] = True
            
    def _initialize_models(self) -> None:
        """設定に基づいてLLMモデルを初期化します"""
        # OpenAI
        if self.config['providers']['openai']['enabled']:
            api_key = self.config['providers']['openai']['api_key']
            if api_key:
                try:
                    model_name = self.config['providers']['openai']['model']
                    self.llms['openai'] = OpenAI(model_name=model_name, api_key=api_key)
                    self.chat_models['openai'] = ChatOpenAI(model_name=model_name, api_key=api_key)
                    logger.info(f"OpenAIモデルを初期化しました: {model_name}")
                except Exception as e:
                    logger.error(f"OpenAIモデルの初期化に失敗しました: {str(e)}")
            else:
                logger.warning("OpenAIのAPIキーが設定されていません")
                
        # Google
        if self.config['providers']['google']['enabled']:
            api_key = self.config['providers']['google']['api_key']
            if api_key:
                try:
                    model_name = self.config['providers']['google']['model']
                    self.llms['google'] = GooglePalm(model_name=model_name, api_key=api_key)
                    self.chat_models['google'] = ChatGooglePalm(model_name=model_name, api_key=api_key)
                    logger.info(f"Googleモデルを初期化しました: {model_name}")
                except Exception as e:
                    logger.error(f"Googleモデルの初期化に失敗しました: {str(e)}")
            else:
                logger.warning("GoogleのAPIキーが設定されていません")
                
        # Anthropic (Claude)
        if self.config['providers']['anthropic']['enabled']:
            api_key = self.config['providers']['anthropic']['api_key']
            if api_key:
                try:
                    model_name = self.config['providers']['anthropic']['model']
                    self.chat_models['anthropic'] = ChatAnthropic(model=model_name, api_key=api_key)
                    logger.info(f"Anthropicモデルを初期化しました: {model_name}")
                except Exception as e:
                    logger.error(f"Anthropicモデルの初期化に失敗しました: {str(e)}")
            else:
                logger.warning("AnthropicのAPIキーが設定されていません")
                
        # ローカルLLM
        if self.config['providers']['local']['enabled'] and LOCAL_LLM_AVAILABLE:
            model_path = self.config['providers']['local']['model_path']
            if model_path and os.path.exists(model_path):
                try:
                    model_type = self.config['providers']['local']['model_type']
                    context_length = self.config['providers']['local']['context_length']
                    
                    if model_type == 'llama':
                        self.llms['local'] = LlamaCpp(
                            model_path=model_path,
                            temperature=0.7,
                            max_tokens=context_length,
                            n_ctx=context_length
                        )
                    else:
                        self.llms['local'] = CTransformers(
                            model=model_path,
                            model_type=model_type,
                            config={'max_new_tokens': 512, 'context_length': context_length}
                        )
                    logger.info(f"ローカルLLMを初期化しました: {model_type} ({model_path})")
                except Exception as e:
                    logger.error(f"ローカルLLMの初期化に失敗しました: {str(e)}")
            else:
                logger.warning(f"ローカルLLMのモデルパスが見つかりません: {model_path}")
                
        # 埋め込みモデルの初期化
        embed_provider = self.config['embeddings']['provider']
        embed_model = self.config['embeddings']['model']
        
        try:
            if embed_provider == 'openai' and self.config['providers']['openai']['enabled']:
                self.embedding_model = OpenAIEmbeddings(
                    model=embed_model,
                    api_key=self.config['providers']['openai']['api_key']
                )
            elif embed_provider == 'google' and self.config['providers']['google']['enabled']:
                self.embedding_model = GooglePalmEmbeddings(
                    model_name=embed_model,
                    api_key=self.config['providers']['google']['api_key']
                )
            elif embed_provider == 'huggingface':
                self.embedding_model = HuggingFaceEmbeddings(model_name=embed_model)
            else:
                # デフォルトのフォールバック
                if 'openai' in self.config['providers'] and self.config['providers']['openai']['enabled']:
                    self.embedding_model = OpenAIEmbeddings(api_key=self.config['providers']['openai']['api_key'])
                    
            if self.embedding_model:
                logger.info(f"埋め込みモデルを初期化しました: {embed_provider} ({embed_model})")
        except Exception as e:
            logger.error(f"埋め込みモデルの初期化に失敗しました: {str(e)}")
                
    def create_knowledge_base(self, documents_dir: str) -> None:
        """
        指定したディレクトリのドキュメントから知識ベースを作成します。
        
        Args:
            documents_dir: ドキュメントファイルが含まれるディレクトリパス
        """
        if not self.embedding_model:
            logger.error("埋め込みモデルが初期化されていないため、知識ベースを作成できません。")
            return
            
        try:
            # ドキュメントローダーの設定
            documents = []
            dir_path = Path(documents_dir)
            
            # テキストファイルの読み込み
            for txt_file in dir_path.glob("**/*.txt"):
                loader = TextLoader(str(txt_file), encoding='utf-8')
                documents.extend(loader.load())
                
            # PDFファイルの読み込み
            for pdf_file in dir_path.glob("**/*.pdf"):
                loader = PyPDFLoader(str(pdf_file))
                documents.extend(loader.load())
                
            if not documents:
                logger.warning(f"指定されたディレクトリ {documents_dir} にドキュメントが見つかりませんでした。")
                return
                
            # ドキュメントの分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            texts = text_splitter.split_documents(documents)
            
            # ベクトルストアの作成
            self.knowledge_base = FAISS.from_documents(texts, self.embedding_model)
            
            logger.info(f"{len(texts)}のテキストチャンクから知識ベースを作成しました。")
            
        except Exception as e:
            logger.error(f"知識ベースの作成に失敗しました: {str(e)}")
            
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[dict]:
        """
        知識ベースを検索して関連情報を取得します。
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位結果の数
            
        Returns:
            関連ドキュメントのリスト
        """
        if not self.knowledge_base:
            logger.error("知識ベースが初期化されていません。")
            return []
            
        try:
            results = self.knowledge_base.similarity_search_with_score(query, k=top_k)
            formatted_results = []
            
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': score
                })
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"知識ベースの検索に失敗しました: {str(e)}")
            return []
            
    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        provider: Optional[str] = None,
        images: Optional[List[str]] = None
    ) -> str:
        """
        プロンプトに基づいてLLMから応答を生成します。
        
        Args:
            prompt: 質問またはプロンプト
            system_message: システムメッセージ（指示）
            provider: 使用するLLMプロバイダー（未指定時はデフォルト）
            images: 画像ファイルパスのリスト（マルチモーダル入力用）
            
        Returns:
            生成された応答テキスト
        """
        if not provider:
            provider = self.config['default_provider']
            
        if provider not in self.chat_models and provider not in self.llms:
            available = list(self.chat_models.keys()) + list(self.llms.keys())
            if not available:
                logger.error("利用可能なLLMプロバイダーがありません。")
                return "エラー: 利用可能なLLMプロバイダーがありません。"
            else:
                provider = available[0]
                logger.warning(f"指定されたプロバイダー {provider} は利用できません。代わりに {provider} を使用します。")
                
        # 知識ベースから関連情報を取得
        context = ""
        if self.knowledge_base:
            relevant_docs = self.search_knowledge_base(prompt)
            if relevant_docs:
                context = "関連情報:\n" + "\n".join([doc['content'] for doc in relevant_docs[:3]])
                
        try:
            # マルチモーダル入力の処理（画像あり）
            if images and provider in ['openai', 'google', 'anthropic']:
                return self._generate_multimodal_response(prompt, system_message, provider, images, context)
            
            # テキストのみの入力
            if provider in self.chat_models:
                # チャットモデルを使用
                messages = []
                
                if system_message:
                    messages.append(SystemMessage(content=system_message))
                    
                if context:
                    messages.append(SystemMessage(content=f"以下の情報を考慮して回答してください:\n{context}"))
                    
                # メモリからの会話履歴を追加
                for message in self.memory.load_memory_variables({})['chat_history']:
                    messages.append(message)
                    
                messages.append(HumanMessage(content=prompt))
                
                response = self.chat_models[provider].predict_messages(messages)
                self.memory.save_context({"input": prompt}, {"output": response.content})
                return response.content
            
            elif provider in self.llms:
                # 標準LLMを使用
                full_prompt = ""
                if system_message:
                    full_prompt += f"{system_message}\n\n"
                    
                if context:
                    full_prompt += f"関連情報:\n{context}\n\n"
                    
                full_prompt += f"質問: {prompt}\n答え:"
                
                response = self.llms[provider].generate([full_prompt])
                return response.generations[0][0].text
                
        except Exception as e:
            logger.error(f"応答生成中にエラーが発生しました: {str(e)}")
            return f"エラー: 応答生成に失敗しました。{str(e)}"
            
    def _generate_multimodal_response(
        self,
        prompt: str,
        system_message: Optional[str],
        provider: str,
        images: List[str],
        context: str
    ) -> str:
        """マルチモーダル（テキスト+画像）応答を生成します"""
        
        if provider == 'openai':
            from openai import OpenAI
            
            client = OpenAI(api_key=self.config['providers']['openai']['api_key'])
            
            # OpenAIのマルチモーダルフォーマットを作成
            messages = []
            
            if system_message:
                messages.append({"role": "system", "content": system_message})
                
            if context:
                messages.append({"role": "system", "content": f"以下の情報を考慮して回答してください:\n{context}"})
                
            # メッセージ内容の準備
            content = [{"type": "text", "text": prompt}]
            
            # 画像の追加
            for img_path in images:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                    
            messages.append({"role": "user", "content": content})
            
            response = client.chat.completions.create(
                model=self.config['providers']['openai']['model'],
                messages=messages
            )
            
            return response.choices[0].message.content
            
        elif provider == 'anthropic':
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.config['providers']['anthropic']['api_key'])
            
            # システムプロンプトを構築
            system_prompt = system_message or ""
            if context:
                system_prompt += f"\n\n以下の情報を考慮して回答してください:\n{context}"
                
            # メッセージ内容の準備
            message = {"text": prompt, "type": "text"}
            multimodal_blocks = [message]
            
            # 画像の追加
            for img_path in images:
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                        
                    multimodal_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    })
                    
            response = client.messages.create(
                model=self.config['providers']['anthropic']['model'],
                system=system_prompt,
                messages=[{"role": "user", "content": multimodal_blocks}],
                max_tokens=2000
            )
            
            return response.content[0].text
            
        elif provider == 'google':
            import google.generativeai as genai
            
            genai.configure(api_key=self.config['providers']['google']['api_key'])
            model = genai.GenerativeModel(self.config['providers']['google']['model'])
            
            # 画像を準備
            image_parts = []
            for img_path in images:
                if os.path.exists(img_path):
                    image_parts.append({
                        'mime_type': 'image/jpeg',
                        'file_path': img_path
                    })
            
            # プロンプトを準備
            full_prompt = ""
            if system_message:
                full_prompt += f"{system_message}\n\n"
                
            if context:
                full_prompt += f"関連情報:\n{context}\n\n"
                
            full_prompt += prompt
            
            response = model.generate_content([full_prompt] + image_parts)
            
            return response.text
            
        else:
            return "このプロバイダーはマルチモーダル入力をサポートしていません。"
            
    def explain_cortical_activation(
        self,
        layer_activations: Dict[str, np.ndarray],
        cortical_outputs: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        大脳皮質モデルの活性化状態とその出力を説明します。
        
        Args:
            layer_activations: 各層の活性化状態
            cortical_outputs: モデルの出力値
            context: 状況に関する追加情報
            
        Returns:
            説明を含む辞書
        """
        # プロンプトを準備
        prompt = "大脳皮質モデルの活性化パターンと出力を分析し、その挙動を説明してください。\n\n"
        
        # 層の活性化情報
        prompt += "各層の活性化パターン:\n"
        for layer_name, activation in layer_activations.items():
            # 活性化の基本統計を計算
            mean_act = np.mean(activation)
            max_act = np.max(activation)
            min_act = np.min(activation)
            std_act = np.std(activation)
            
            prompt += f"- {layer_name}: 平均={mean_act:.4f}, 最大={max_act:.4f}, 最小={min_act:.4f}, 標準偏差={std_act:.4f}\n"
            
        # 出力情報
        prompt += "\nモデル出力:\n"
        for i, output in enumerate(cortical_outputs):
            prompt += f"出力{i}: {output:.4f}\n"
            
        # コンテキスト情報があれば追加
        if context:
            prompt += "\n状況情報:\n"
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    prompt += f"- {key}: {value}\n"
                    
        # システムメッセージ
        system_message = """
        あなたは高度な神経科学の専門家として、大脳皮質モデルの挙動を分析し説明します。
        各層の活性化パターンとモデル出力から、モデルが何を知覚し、どのような意思決定を行っているかを推測してください。
        説明は科学的に正確で、かつ一般の人にも理解できるように簡潔にしてください。
        """
        
        # 応答を生成
        response = self.generate_response(prompt, system_message)
        
        return {
            'explanation': response,
            'analysis': {
                'layer_stats': {layer: {
                    'mean': float(np.mean(act)),
                    'max': float(np.max(act)),
                    'min': float(np.min(act)),
                    'std': float(np.std(act))
                } for layer, act in layer_activations.items()},
                'outputs': [float(o) for o in cortical_outputs]
            }
        }
        
    def explain_decision(
        self,
        input_data: Dict[str, Any],
        output_action: np.ndarray,
        layer_activations: Dict[str, Dict[str, Any]],
        neurotransmitter_levels: Dict[str, float],
        images: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        ロボットの意思決定プロセスを説明します。
        
        Args:
            input_data: 入力センサーデータ
            output_action: モデルが選択した行動
            layer_activations: 各層の活性化状態
            neurotransmitter_levels: 神経伝達物質レベル
            images: 視覚入力画像のパスリスト（オプション）
            
        Returns:
            説明テキストを含む辞書
        """
        # プロンプト構築
        prompt = "ロボットの意思決定プロセスを詳細に説明してください。\n\n"
        
        # 入力情報
        prompt += "入力センサーデータ:\n"
        for key, value in input_data.items():
            if isinstance(value, np.ndarray):
                summary = f"形状={value.shape}, 平均={np.mean(value):.4f}, 最大={np.max(value):.4f}"
                prompt += f"- {key}: {summary}\n"
            elif isinstance(value, (int, float, str, bool)):
                prompt += f"- {key}: {value}\n"
                
        # 出力行動
        prompt += "\n選択された行動:\n"
        for i, action_value in enumerate(output_action):
            prompt += f"行動{i}: {action_value:.4f}\n"
            
        # 神経伝達物質レベル
        prompt += "\n神経伝達物質レベル:\n"
        for nt, level in neurotransmitter_levels.items():
            prompt += f"- {nt}: {level:.2f}\n"
            
        # システムメッセージ
        system_message = """
        あなたは高度な説明可能AIシステムとして、ロボットの意思決定プロセスを説明します。
        入力データ、神経ネットワークの活性化パターン、神経伝達物質レベル、および出力行動に基づいて、
        ロボットがなぜこの行動を選択したのかを説明してください。
        説明は論理的で、かつ一般の人にも理解できるように簡潔にしてください。
        """
        
        # 画像があればマルチモーダル応答を生成
        response = self.generate_response(prompt, system_message, images=images)
        
        # 説明を構造化
        return {
            'detailed_explanation': response,
            'summary': self._generate_summary(response),
            'confidence': self._estimate_confidence(layer_activations, output_action)
        }
        
    def _generate_summary(self, detailed_explanation: str) -> str:
        """詳細な説明から要約を生成します"""
        summary_prompt = f"以下の説明を3文以内で要約してください:\n\n{detailed_explanation}"
        return self.generate_response(summary_prompt, "短く簡潔に要約してください。")
        
    def _estimate_confidence(
        self,
        layer_activations: Dict[str, Dict[str, Any]],
        output_action: np.ndarray
    ) -> float:
        """
        モデルの決定に対する信頼度を推定します。
        
        Returns:
            信頼度スコア (0.0〜1.0)
        """
        # 出力の明確さ（最大値と平均値の差）
        output_clarity = float(np.max(output_action) - np.mean(output_action))
        
        # 層5（主要出力層）の活性化の強さ
        layer5_strength = 0.5  # デフォルト値
        if 'layer5' in layer_activations:
            layer5_stats = layer_activations['layer5']
            if isinstance(layer5_stats, dict) and 'mean' in layer5_stats:
                layer5_strength = min(1.0, layer5_stats['mean'] * 2)
                
        # 最終層の活性化の一貫性（標準偏差の低さ）
        activation_consistency = 0.5  # デフォルト値
        for layer_name in ['layer5', 'layer6', 'output']:
            if layer_name in layer_activations:
                layer_stats = layer_activations[layer_name]
                if isinstance(layer_stats, dict) and 'std' in layer_stats:
                    # 標準偏差の逆数（低いほど一貫性が高い）
                    consistency = 1.0 - min(1.0, layer_stats['std'])
                    activation_consistency = max(activation_consistency, consistency)
                    
        # 総合信頼度スコア
        confidence = (output_clarity * 0.4 + layer5_strength * 0.3 + activation_consistency * 0.3)
        return min(1.0, max(0.0, confidence))

    def save_config(self, config_path: str) -> bool:
        """
        現在の設定をJSONファイルに保存します。
        APIキーは保存されません。
        
        Args:
            config_path: 保存先のファイルパス
            
        Returns:
            保存が成功したかどうか
        """
        # APIキーを除外した設定のコピーを作成
        safe_config = self.config.copy()
        
        # APIキーを削除
        for provider in safe_config['providers']:
            if 'api_key' in safe_config['providers'][provider]:
                safe_config['providers'][provider]['api_key'] = None
                
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(safe_config, f, indent=2, ensure_ascii=False)
            logger.info(f"設定を {config_path} に保存しました")
            return True
        except Exception as e:
            logger.error(f"設定の保存に失敗しました: {str(e)}")
            return False
            
    def reset_memory(self) -> None:
        """会話履歴をリセットします"""
        self.memory.clear()
        logger.info("会話履歴をリセットしました")


def create_multimodal_llm(config_path: Optional[str] = None) -> MultimodalLLMSystem:
    """
    マルチモーダルLLMシステムのインスタンスを作成するファクトリ関数
    
    Args:
        config_path: 設定ファイルへのパス（オプション）
        
    Returns:
        初期化されたMultimodalLLMSystemインスタンス
    """
    return MultimodalLLMSystem(config_path) 