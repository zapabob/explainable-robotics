"""
説明生成ユーティリティモジュール

このモジュールはニューラルネットワークモデルの決定プロセスを説明するための
ユーティリティ関数とツールを提供します。
"""

import json
import os
import datetime
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np
import torch
from ..utils.logging import get_logger

# ロガーの設定
logger = get_logger(__name__)

class ExplanationGenerator:
    """
    モデルの決定プロセスの説明を生成するクラス
    
    このクラスは、大脳皮質層モデルの各層の活性化パターンを分析し、
    自然言語での説明を生成します。
    """
    
    def __init__(self, language: str = 'ja'):
        """
        初期化
        
        Args:
            language: 説明言語 ('ja'または'en')
        """
        self.language = language
        
        # 言語別の説明テンプレート
        self.templates = {
            'ja': {
                'layer_activity': "層{layer_name}は{activity_description}",
                'confidence_high': "高い確信度（{confidence:.1%}）で",
                'confidence_medium': "中程度の確信度（{confidence:.1%}）で",
                'confidence_low': "低い確信度（{confidence:.1%}）で",
                'action_forward': "前進する",
                'action_backward': "後退する",
                'action_stop': "停止する",
                'action_turn': "{direction}方向に{degree:.1f}度回転する",
                'reasoning': "これは{reason}ためです。"
            },
            'en': {
                'layer_activity': "Layer {layer_name} shows {activity_description}",
                'confidence_high': "with high confidence ({confidence:.1%})",
                'confidence_medium': "with moderate confidence ({confidence:.1%})",
                'confidence_low': "with low confidence ({confidence:.1%})",
                'action_forward': "moving forward",
                'action_backward': "moving backward",
                'action_stop': "stopping",
                'action_turn': "turning {direction} by {degree:.1f} degrees",
                'reasoning': "This is because {reason}."
            }
        }
    
    def generate_explanation(
        self, 
        model_output: Union[np.ndarray, torch.Tensor],
        layer_activations: Dict[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        モデル出力の説明を生成
        
        Args:
            model_output: モデルの出力値
            layer_activations: 各層の活性化情報
            context: 追加のコンテキスト情報
            
        Returns:
            説明情報を含む辞書
        """
        # 出力を適切な形式に変換
        if isinstance(model_output, torch.Tensor):
            output = model_output.detach().cpu().numpy()
        else:
            output = model_output
            
        # 現在の言語のテンプレート
        tmpl = self.templates.get(self.language, self.templates['ja'])
        
        # 出力値の解釈
        motor_actions = self._interpret_motor_output(output)
        
        # 確信度のフォーマット
        confidence = self._calculate_confidence(layer_activations)
        if confidence > 0.7:
            confidence_text = tmpl['confidence_high'].format(confidence=confidence)
        elif confidence > 0.4:
            confidence_text = tmpl['confidence_medium'].format(confidence=confidence)
        else:
            confidence_text = tmpl['confidence_low'].format(confidence=confidence)
        
        # 行動の理由
        reasoning = self._determine_reasoning(layer_activations, output, context)
        reason_text = tmpl['reasoning'].format(reason=reasoning)
        
        # 層ごとの説明
        layer_explanations = {}
        for layer_name, activation_info in layer_activations.items():
            if 'interpretation' in activation_info:
                layer_explanations[layer_name] = activation_info['interpretation']
        
        # 自然言語での説明文を生成
        action_text = self._format_action_text(motor_actions, tmpl)
        narrative = f"{action_text} {confidence_text}。{reason_text}"
        
        # 最終的な説明を返す
        explanation = {
            'narrative': narrative,
            'motor_actions': motor_actions,
            'confidence': confidence,
            'reasoning': reasoning,
            'layer_explanations': layer_explanations,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        logger.debug(f"説明を生成しました: {narrative}")
        return explanation
    
    def _interpret_motor_output(self, output: np.ndarray) -> Dict[str, float]:
        """
        モーター出力の解釈
        
        Args:
            output: モデル出力値
            
        Returns:
            モーター動作を表す辞書
        """
        # 出力形式に応じた解釈（単純化のため、1次元または2次元と仮定）
        output_mean = np.mean(output, axis=0) if output.ndim > 1 else output
        
        if output.size == 1:
            # 単一値の場合（前進/後退）
            value = float(output_mean)
            return {'forward': value}
        
        elif output.size == 2:
            # 2値の場合（前進/後退と回転）
            return {
                'forward': float(output_mean[0]),
                'turn': float(output_mean[1])
            }
        
        elif output.size == 3:
            # 3値の場合（x, y, z方向）
            return {
                'forward': float(output_mean[0]),  # 前後
                'lateral': float(output_mean[1]),  # 左右
                'vertical': float(output_mean[2])  # 上下
            }
        
        else:
            # 多次元の場合は主要な値のみを抽出
            result = {}
            for i, val in enumerate(output_mean[:5]):  # 最初の5つの値のみ
                result[f'motor_{i}'] = float(val)
            return result
    
    def _calculate_confidence(self, layer_activations: Dict[str, Dict[str, Any]]) -> float:
        """
        説明の確信度を計算
        
        Args:
            layer_activations: 各層の活性化情報
            
        Returns:
            確信度（0〜1）
        """
        # 出力層（L5）の情報に基づく確信度
        l5_info = layer_activations.get('L5', {})
        stats = l5_info.get('statistics', {})
        
        # 出力の標準偏差が低いほど確信度が高い
        std = stats.get('std', 0.5)  # デフォルト値
        
        # 確信度の計算（標準偏差が0.5以上なら確信度0、0なら確信度1）
        confidence = max(0.0, 1.0 - (std * 2))
        return float(confidence)
    
    def _determine_reasoning(
        self,
        layer_activations: Dict[str, Dict[str, Any]],
        output: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        行動の理由を決定
        
        Args:
            layer_activations: 各層の活性化情報
            output: モデル出力
            context: 追加のコンテキスト情報
            
        Returns:
            理由の説明文
        """
        # 基本的な理由説明（言語に応じて調整）
        is_japanese = self.language == 'ja'
        
        # 入力層（L1）の情報から入力パターンを解釈
        l1_info = layer_activations.get('L1', {})
        l1_interp = l1_info.get('interpretation', '')
        
        # 出力層（L5）の情報から出力パターンを解釈
        l5_info = layer_activations.get('L5', {})
        l5_interp = l5_info.get('interpretation', '')
        
        # コンテキスト情報から理由を構築
        if context is not None:
            # 障害物情報
            obstacles = context.get('obstacles', False)
            if obstacles:
                return "障害物を検出した" if is_japanese else "obstacles were detected"
            
            # 目標地点情報
            target = context.get('target_position')
            if target is not None:
                return "目標地点に向かっている" if is_japanese else "moving towards the target position"
        
        # 入力と出力の情報から理由を推測
        if "多様な感覚入力" in l1_interp or "diverse sensory input" in l1_interp:
            return "複雑な環境に対応している" if is_japanese else "adapting to a complex environment"
        
        if "一貫した感覚入力" in l1_interp or "consistent sensory input" in l1_interp:
            return "安定した環境を走行している" if is_japanese else "traveling in a stable environment"
        
        # 運動コマンドから理由を推測
        if "強い" in l5_interp or "strong" in l5_interp:
            return "明確な方向指示を受けた" if is_japanese else "received clear directional instructions"
        
        if "バランス" in l5_interp or "balance" in l5_interp:
            return "安定状態を維持している" if is_japanese else "maintaining a stable state"
        
        # デフォルトの理由
        return "現在の環境と入力状態に基づいて判断した" if is_japanese else "based on the current environment and input state"
    
    def _format_action_text(self, actions: Dict[str, float], tmpl: Dict[str, str]) -> str:
        """
        行動テキストをフォーマット
        
        Args:
            actions: 行動辞書
            tmpl: 言語テンプレート
            
        Returns:
            フォーマットされた行動テキスト
        """
        # 前進/後退の処理
        forward = actions.get('forward', 0.0)
        
        if abs(forward) < 0.1:
            action_text = tmpl['action_stop']
        elif forward > 0:
            action_text = tmpl['action_forward']
        else:
            action_text = tmpl['action_backward']
        
        # 回転がある場合
        turn = actions.get('turn', 0.0)
        if abs(turn) > 0.1:
            direction = "右" if turn > 0 else "左"
            if self.language != 'ja':
                direction = "right" if turn > 0 else "left"
            
            degree = abs(turn) * 90  # -1〜1の値を0〜90度に変換
            
            # 前進/後退がある場合は、「〜しながら」というテキストを追加
            if abs(forward) > 0.1:
                action_text += "ながら" if self.language == 'ja' else " while"
            
            action_text += " " + tmpl['action_turn'].format(direction=direction, degree=degree)
        
        return action_text

def generate_explanation(
    model_output: Union[np.ndarray, torch.Tensor],
    layer_activations: Dict[str, Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    language: str = 'ja'
) -> Dict[str, Any]:
    """
    モデル出力の説明を生成する便利関数
    
    Args:
        model_output: モデルの出力値
        layer_activations: 各層の活性化情報
        context: 追加のコンテキスト情報
        language: 説明言語 ('ja'または'en')
        
    Returns:
        説明情報を含む辞書
    """
    generator = ExplanationGenerator(language=language)
    return generator.generate_explanation(model_output, layer_activations, context)

def save_explanation(explanation: Dict[str, Any], filename: str) -> None:
    """
    説明情報をJSONファイルに保存
    
    Args:
        explanation: 説明情報
        filename: 保存先のファイル名
    """
    # ディレクトリが存在しない場合は作成
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    # 説明情報を保存
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(explanation, f, indent=2, ensure_ascii=False)
    
    logger.info(f"説明情報を保存しました: {filename}")

def load_explanation(filename: str) -> Dict[str, Any]:
    """
    保存された説明情報を読み込み
    
    Args:
        filename: 読み込むファイル名
        
    Returns:
        説明情報
    """
    with open(filename, 'r', encoding='utf-8') as f:
        explanation = json.load(f)
    
    logger.info(f"説明情報を読み込みました: {filename}")
    return explanation 