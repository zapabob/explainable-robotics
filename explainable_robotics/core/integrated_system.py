import os
import time
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import json
from pathlib import Path
import threading

# 内部モジュールのインポート
from ..cortical.neurotransmitters import NeurotransmitterSystem, GlutamateSystem, GABASystem, DrugSystem
from ..llm.langchain_integration import MultimodalLLMSystem, create_multimodal_llm
from ..genesis.robot_interface import GenesisRobotInterface, create_robot_interface

logger = logging.getLogger(__name__)

class IntegratedSystem:
    """
    神経伝達物質システム、LangChainマルチモーダルLLM、Genesisロボットインターフェースを
    統合した総合システム。生物学的に妥当なモデルでヒューマノイドロボットを制御します。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # 設定
        self.config = {
            'system': {
                'data_dir': './data',
                'knowledge_dir': './data/knowledge',
                'log_dir': './logs',
                'explanation_dir': './data/explanations'
            },
            'neurotransmitters': {
                'default_levels': {
                    'acetylcholine': 0.5,
                    'dopamine': 0.5,
                    'serotonin': 0.5,
                    'noradrenaline': 0.5,
                    'glutamate': 0.5,
                    'gaba': 0.5
                }
            },
            'cortical_model': {
                'layers': ['layer1', 'layer2_3', 'layer4', 'layer5', 'layer6'],
                'input_dim': 100,
                'output_dim': 22  # ヒューマノイドロボットのモーター数に合わせる
            },
            'robot_interface': {
                'simulation_mode': True
            },
            'llm': {
                'default_provider': 'openai'
            }
        }
        
        # 設定ファイルからのロード
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # マージ
                    self._merge_config(loaded_config)
            except Exception as e:
                logger.error(f"設定ファイルの読み込みに失敗しました: {str(e)}")
                
        # サブシステムの設定ファイルパス
        self.nt_config_path = None
        self.llm_config_path = None
        self.robot_config_path = None
        
        if config_path:
            config_dir = os.path.dirname(config_path)
            self.nt_config_path = os.path.join(config_dir, 'neurotransmitters_config.json')
            self.llm_config_path = os.path.join(config_dir, 'llm_config.json')
            self.robot_config_path = os.path.join(config_dir, 'robot_config.json')
            
        # 必要なディレクトリの作成
        self._create_directories()
        
        # サブシステムの初期化
        self.nt_system = NeurotransmitterSystem()
        self.glutamate_system = GlutamateSystem()
        self.gaba_system = GABASystem()
        self.drug_system = DrugSystem(self.nt_system)
        
        self.llm_system = create_multimodal_llm(self.llm_config_path)
        self.robot_interface = create_robot_interface(self.robot_config_path)
        
        # 内部状態
        self.running = False
        self.sensor_data = {}
        self.cortical_activation = {}
        self.motor_commands = {}
        self.last_explanation = {}
        
        # 中枢神経用薬デフォルトレベルの設定
        for nt, level in self.config['neurotransmitters']['default_levels'].items():
            self.nt_system.set_level(nt, level)
            
        # 制御ループのスレッド
        self.control_thread = None
        
        logger.info("統合システムを初期化しました")
        
    def _merge_config(self, new_config: Dict) -> None:
        """設定を再帰的にマージします"""
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    self._merge_config_recursive(self.config[key], value)
                else:
                    self.config[key] = value
                    
    def _merge_config_recursive(self, target: Dict, source: Dict) -> None:
        """設定を再帰的にマージします（内部ヘルパー）"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(target[key], value)
            else:
                target[key] = value
                
    def _create_directories(self) -> None:
        """必要なディレクトリを作成します"""
        directories = [
            self.config['system']['data_dir'],
            self.config['system']['knowledge_dir'],
            self.config['system']['log_dir'],
            self.config['system']['explanation_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def initialize_knowledge_base(self, documents_dir: Optional[str] = None) -> None:
        """
        LLMの知識ベースを初期化します。
        
        Args:
            documents_dir: 知識ベース用ドキュメントのディレクトリ（指定がなければデフォルト）
        """
        if documents_dir is None:
            documents_dir = self.config['system']['knowledge_dir']
            
        logger.info(f"知識ベースを初期化します: {documents_dir}")
        self.llm_system.create_knowledge_base(documents_dir)
        
    def apply_drug(self, drug_name: str, dose: float = 1.0) -> None:
        """
        中枢神経作用薬を適用し、神経伝達物質レベルに影響を与えます。
        
        Args:
            drug_name: 薬物の名前
            dose: 投与量（相対値）
        """
        logger.info(f"薬物を適用します: {drug_name} (投与量: {dose:.2f})")
        self.drug_system.apply_drug(drug_name, dose)
        
        # 薬効の説明を生成
        effects = self.drug_system.describe_drug_effects()
        logger.info(f"薬物効果: {effects}")
        
    def adjust_neurotransmitter(self, transmitter_type: str, level: float) -> None:
        """
        特定の神経伝達物質レベルを直接調整します。
        
        Args:
            transmitter_type: 神経伝達物質のタイプ
            level: 設定するレベル（0.0〜1.0）
        """
        logger.info(f"神経伝達物質を調整します: {transmitter_type} -> {level:.2f}")
        self.nt_system.set_level(transmitter_type, level)
        
        # グルタミン酸とGABAの特別な処理
        if transmitter_type == 'glutamate':
            self.glutamate_system.modulate(level - self.glutamate_system.current_level)
        elif transmitter_type == 'gaba':
            self.gaba_system.modulate(level - self.gaba_system.current_level)
            
    def adjust_receptor_sensitivity(
        self,
        transmitter_type: str,
        receptor_changes: Dict[str, float]
    ) -> None:
        """
        特定の神経伝達物質受容体の感受性を調整します。
        
        Args:
            transmitter_type: 神経伝達物質のタイプ
            receptor_changes: 受容体名と変化量のマッピング
        """
        if transmitter_type == 'glutamate':
            logger.info(f"グルタミン酸受容体感受性を調整します: {receptor_changes}")
            self.glutamate_system.modulate(0, receptor_changes)
        elif transmitter_type == 'gaba':
            logger.info(f"GABA受容体感受性を調整します: {receptor_changes}")
            self.gaba_system.modulate(0, receptor_changes)
        else:
            logger.warning(f"受容体調整は {transmitter_type} ではサポートされていません")
            
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """
        センサーデータを処理し、皮質モデル出力を生成します。
        実際のシステムでは、ここに神経科学的に妥当なモデルが実装されます。
        
        Args:
            sensor_data: ロボットのセンサーデータ
            
        Returns:
            モーター制御用の出力ベクトル
        """
        # 神経伝達物質の変調係数を取得
        modulation = self.nt_system.get_modulation_factors()
        
        # 単純化された皮質活性化のシミュレーション
        # 実際のシステムでは、より複雑な皮質モデルを使用
        cortical_activation = {}
        
        # 層1: 感覚入力層
        input_sensitivity = modulation['input_sensitivity']
        cortical_activation['layer1'] = self._simulate_layer1_activation(sensor_data, input_sensitivity)
        
        # 層2/3: 横方向接続と特徴統合
        lateral_inhibition = self.gaba_system.get_effect_on_layer('layer2_3').get('lateral_inhibition', 0.5)
        cortical_activation['layer2_3'] = self._simulate_layer2_3_activation(
            cortical_activation['layer1'],
            lateral_inhibition,
            modulation['synaptic_plasticity']
        )
        
        # 層4: 視床からの入力
        sensory_gain = self.glutamate_system.get_effect_on_layer('layer4').get('sensory_gain', 0.5)
        noise_reduction = self.gaba_system.get_effect_on_layer('layer4').get('noise_reduction', 0.5)
        cortical_activation['layer4'] = self._simulate_layer4_activation(
            sensor_data,
            sensory_gain,
            noise_reduction
        )
        
        # 層5: 主な出力層
        activation_strength = modulation['activation_strength']
        cortical_activation['layer5'] = self._simulate_layer5_activation(
            cortical_activation['layer2_3'],
            cortical_activation['layer4'],
            activation_strength
        )
        
        # 層6: 視床へのフィードバック層
        e_i_balance = modulation['e_i_balance']
        cortical_activation['layer6'] = self._simulate_layer6_activation(
            cortical_activation['layer5'],
            e_i_balance
        )
        
        # 最終的なモーター出力
        output_dim = self.config['cortical_model']['output_dim']
        output = self._generate_motor_output(
            cortical_activation,
            modulation,
            output_dim
        )
        
        # 内部状態を更新
        self.sensor_data = sensor_data.copy()
        self.cortical_activation = cortical_activation
        
        return output
        
    def _simulate_layer1_activation(
        self,
        sensor_data: Dict[str, Any],
        input_sensitivity: float
    ) -> np.ndarray:
        """層1の活性化をシミュレート（感覚入力）"""
        # センサーデータから入力ベクトルを作成
        input_vector = np.zeros(self.config['cortical_model']['input_dim'])
        
        # IMUデータを抽出（存在すれば）
        if 'imu' in sensor_data:
            if len(input_vector) >= 6:
                # 最初の3要素にジャイロデータ
                input_vector[0:3] = sensor_data['imu']['gyro'] * input_sensitivity
                # 次の3要素に加速度データ
                input_vector[3:6] = sensor_data['imu']['accel'] * input_sensitivity
                
        # 関節センサーデータを抽出（存在すれば）
        if 'joints' in sensor_data:
            joints = sensor_data['joints']
            i = 6  # IMUデータの後から開始
            
            for joint_name, joint_data in joints.items():
                if i < len(input_vector) - 2:
                    input_vector[i] = joint_data['position'] * input_sensitivity
                    input_vector[i+1] = joint_data['velocity'] * input_sensitivity
                    input_vector[i+2] = joint_data['load'] * input_sensitivity
                    i += 3
                else:
                    break
                    
        # 力センサーデータを抽出（存在すれば）
        if 'force' in sensor_data:
            force = sensor_data['force']
            i = min(i, len(input_vector) - 10)  # 最大10要素使用
            
            if 'left' in force and i < len(input_vector) - 4:
                input_vector[i:i+3] = force['left']['force'] * input_sensitivity
                input_vector[i+3:i+5] = force['left']['center_of_pressure'] * input_sensitivity
                i += 5
                
            if 'right' in force and i < len(input_vector) - 4:
                input_vector[i:i+3] = force['right']['force'] * input_sensitivity
                input_vector[i+3:i+5] = force['right']['center_of_pressure'] * input_sensitivity
                
        return input_vector
        
    def _simulate_layer2_3_activation(
        self,
        layer1_output: np.ndarray,
        lateral_inhibition: float,
        synaptic_plasticity: float
    ) -> np.ndarray:
        """層2/3の活性化をシミュレート（横方向接続と特徴統合）"""
        # シンプルな非線形変換＋側方抑制をシミュレート
        output = np.tanh(layer1_output * synaptic_plasticity)
        
        # 側方抑制（最大応答を強化し、他を抑制）
        if len(output) > 0:
            max_idx = np.argmax(np.abs(output))
            mask = np.ones_like(output) * (1 - lateral_inhibition)
            mask[max_idx] = 1.0
            output = output * mask
            
        return output
        
    def _simulate_layer4_activation(
        self,
        sensor_data: Dict[str, Any],
        sensory_gain: float,
        noise_reduction: float
    ) -> np.ndarray:
        """層4の活性化をシミュレート（視床からの入力）"""
        # 単純化のため、センサーデータを直接使用
        output_size = self.config['cortical_model']['input_dim'] // 2
        output = np.zeros(output_size)
        
        # カメラデータを処理（存在すれば）
        if 'camera' in sensor_data and isinstance(sensor_data['camera'], np.ndarray):
            # 画像の平均輝度などの簡単な特徴を抽出
            camera_data = sensor_data['camera']
            if len(camera_data.shape) == 3:  # (height, width, channels)
                # グレースケールに変換
                if camera_data.shape[2] == 3:  # RGB
                    gray = np.mean(camera_data, axis=2)
                else:
                    gray = camera_data[:, :, 0]
                    
                # 画像を小さなブロックに分割し、各ブロックの平均をとる
                h, w = gray.shape
                block_h, block_w = max(1, h // 4), max(1, w // 4)
                block_means = []
                
                for i in range(0, h, block_h):
                    for j in range(0, w, block_w):
                        block = gray[i:min(i+block_h, h), j:min(j+block_w, w)]
                        block_means.append(np.mean(block))
                        
                # 出力ベクトルの前半に視覚特徴を設定
                visual_features = np.array(block_means) * sensory_gain
                output[:min(len(visual_features), output_size//2)] = visual_features[:output_size//2]
        
        # IMUと力センサーデータから特徴を抽出
        if 'imu' in sensor_data:
            imu_features = np.concatenate([
                sensor_data['imu']['gyro'],
                sensor_data['imu']['accel']
            ]) * sensory_gain
            
            # ノイズ低減処理
            imu_features = imu_features * noise_reduction
            
            # 出力ベクトルの後半に設定
            start_idx = output_size // 2
            end_idx = min(start_idx + len(imu_features), output_size)
            output[start_idx:end_idx] = imu_features[:end_idx-start_idx]
            
        return output
        
    def _simulate_layer5_activation(
        self,
        layer2_3_output: np.ndarray,
        layer4_output: np.ndarray,
        activation_strength: float
    ) -> np.ndarray:
        """層5の活性化をシミュレート（主な出力層）"""
        # 層2/3と層4からの入力を統合
        l2_3_size = len(layer2_3_output)
        l4_size = len(layer4_output)
        
        # 出力サイズは層2/3と層4の平均
        output_size = (l2_3_size + l4_size) // 3
        output = np.zeros(output_size)
        
        # 層2/3と層4からの入力を重み付けして統合
        for i in range(output_size):
            # 層2/3からの入力（利用可能な範囲でインデックス付け）
            if i < l2_3_size:
                output[i] += layer2_3_output[i] * 0.6  # 層2/3からの影響は60%
                
            # 層4からの入力（利用可能な範囲でインデックス付け）
            if i < l4_size:
                output[i] += layer4_output[i] * 0.4  # 層4からの影響は40%
                
        # 活性化強度による全体的なスケーリング
        output = np.tanh(output * activation_strength)
        
        # グルタミン酸による興奮性効果を追加
        excitation = self.glutamate_system.get_effect_on_layer('layer5').get('excitation', 0.5)
        output = output * (0.5 + 0.5 * excitation)
        
        # GABAによる抑制効果を追加
        inhibition = self.gaba_system.get_effect_on_layer('layer5').get('inhibition', 0.5)
        # 抑制は値をゼロに近づける効果がある
        output = output * (1.0 - 0.5 * inhibition)
        
        return output
        
    def _simulate_layer6_activation(
        self,
        layer5_output: np.ndarray,
        e_i_balance: float
    ) -> np.ndarray:
        """層6の活性化をシミュレート（視床へのフィードバック層）"""
        # 層5からの入力を変換
        output_size = len(layer5_output) // 2
        output = np.zeros(output_size)
        
        # 興奮/抑制バランスに基づいて変換
        # e_i_balance > 1 は興奮が優位、< 1 は抑制が優位
        for i in range(output_size):
            idx = min(i * 2, len(layer5_output) - 1)
            if e_i_balance > 1.0:
                # 興奮が優位な場合、活性化を強化
                output[i] = layer5_output[idx] * e_i_balance
            else:
                # 抑制が優位な場合、活性化を減衰
                output[i] = layer5_output[idx] * e_i_balance
                
        # 非線形化
        output = np.tanh(output)
        
        return output
        
    def _generate_motor_output(
        self,
        cortical_activation: Dict[str, np.ndarray],
        modulation: Dict[str, float],
        output_dim: int
    ) -> np.ndarray:
        """皮質活性化からモーター出力を生成"""
        # 層5と層6の出力を使用して最終的なモーター制御を生成
        layer5_output = cortical_activation['layer5']
        layer6_output = cortical_activation['layer6']
        
        # 出力の次元数を調整
        motor_output = np.zeros(output_dim)
        
        # 層5からの直接的な運動指令
        l5_size = len(layer5_output)
        l6_size = len(layer6_output)
        
        # 出力ベクトルに層5と層6の出力を割り当て
        for i in range(output_dim):
            if i < l5_size:
                motor_output[i] = layer5_output[i] * 0.7  # 層5からの影響は70%
                
            if i < l6_size:
                motor_output[i] += layer6_output[i % l6_size] * 0.3  # 層6からの影響は30%
                
        # 行動変動性を適用（ドーパミンレベルに依存）
        action_variability = modulation['action_variability']
        if action_variability > 0:
            # ノイズを追加（変動性に比例）
            noise = np.random.normal(0, action_variability * 0.2, output_dim)
            motor_output += noise
            
        # 応答速度を適用（ノルアドレナリンレベルに依存）
        response_speed = modulation['response_speed']
        if response_speed < 1.0 and hasattr(self, 'last_motor_output'):
            # 前回の出力との線形補間
            motor_output = self.last_motor_output * (1 - response_speed) + motor_output * response_speed
            
        # 範囲を[-1, 1]に制限
        motor_output = np.clip(motor_output, -1.0, 1.0)
        
        # 最後の出力を保存
        self.last_motor_output = motor_output.copy()
        
        return motor_output
        
    def generate_behavior_explanation(self) -> Dict[str, Any]:
        """
        ロボットの行動に関する説明を生成します。
        
        Returns:
            説明テキストと関連データを含む辞書
        """
        if not self.sensor_data or not self.cortical_activation:
            logger.warning("説明を生成するためのデータがありません")
            return {"explanation": "行動データがありません"}
            
        try:
            # 神経伝達物質レベルを取得
            nt_levels = {
                name: self.nt_system.get_level(name)
                for name in ['acetylcholine', 'dopamine', 'serotonin', 'noradrenaline', 'glutamate', 'gaba']
            }
            
            # 層の活性化データをフォーマット
            layer_activations = {
                layer_name: {
                    'mean': float(np.mean(activation)),
                    'max': float(np.max(activation)),
                    'min': float(np.min(activation)),
                    'std': float(np.std(activation))
                }
                for layer_name, activation in self.cortical_activation.items()
            }
            
            # LLMで説明を生成
            explanation = self.llm_system.explain_decision(
                self.sensor_data,
                self.last_motor_output if hasattr(self, 'last_motor_output') else np.zeros(self.config['cortical_model']['output_dim']),
                layer_activations,
                nt_levels
            )
            
            # 説明を保存
            self.last_explanation = explanation
            
            # 説明をファイルに保存
            self._save_explanation(explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"説明の生成中にエラーが発生しました: {str(e)}")
            return {"explanation": f"説明の生成に失敗しました: {str(e)}"}
            
    def _save_explanation(self, explanation: Dict[str, Any]) -> None:
        """説明をファイルに保存します"""
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            explanation_dir = self.config['system']['explanation_dir']
            
            # 詳細な説明テキスト
            if 'detailed_explanation' in explanation:
                text_path = os.path.join(explanation_dir, f"explanation_{timestamp}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(explanation['detailed_explanation'])
                    
            # メタデータとデータをJSONに保存
            data = {
                'timestamp': timestamp,
                'neurotransmitter_levels': {
                    name: self.nt_system.get_level(name)
                    for name in ['acetylcholine', 'dopamine', 'serotonin', 'noradrenaline', 'glutamate', 'gaba']
                },
                'active_drugs': self.nt_system.get_active_drugs(),
                'summary': explanation.get('summary', ''),
                'confidence': explanation.get('confidence', 0.0)
            }
            
            json_path = os.path.join(explanation_dir, f"explanation_data_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"説明の保存中にエラーが発生しました: {str(e)}")
            
    def control_loop(self) -> None:
        """コントロールループの内部実装"""
        logger.info("制御ループを開始します")
        
        try:
            while self.running:
                # センサーデータを読み取り
                sensor_data = self.robot_interface.read_sensors()
                
                # センサーデータを処理し、皮質モデル出力を生成
                cortical_output = self.process_sensor_data(sensor_data)
                
                # モーターコマンドに変換して送信
                motor_commands = self.robot_interface.convert_cortical_output_to_motor_commands(cortical_output)
                self.robot_interface.send_motor_commands(motor_commands)
                self.motor_commands = motor_commands
                
                # 薬物代謝をシミュレート
                self.drug_system.metabolism_step()
                
                # 少し待機
                time.sleep(0.01)
                
        except Exception as e:
            logger.error(f"制御ループでエラーが発生しました: {str(e)}")
        finally:
            logger.info("制御ループを終了します")
            
    def start(self) -> None:
        """システムの実行を開始します"""
        if self.running:
            logger.warning("システムはすでに実行中です")
            return
            
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        logger.info("システムを開始しました")
        
    def stop(self) -> None:
        """システムの実行を停止します"""
        if not self.running:
            logger.warning("システムは実行されていません")
            return
            
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
            
        # ロボットを安全にシャットダウン
        self.robot_interface.shutdown()
        
        logger.info("システムを停止しました")
        
    def execute_gesture(self, gesture_name: str) -> bool:
        """
        事前定義されたジェスチャーを実行します。
        
        Args:
            gesture_name: 実行するジェスチャーの名前
            
        Returns:
            ジェスチャーの実行が成功したかどうか
        """
        return self.robot_interface.execute_gesture(gesture_name)
        
    def explain_current_state(self) -> Dict[str, Any]:
        """
        システムの現在の状態を説明します。
        
        Returns:
            システム状態の説明を含む辞書
        """
        # 神経伝達物質の状態を取得
        nt_state = self.nt_system.describe_current_state()
        
        # 薬物効果を取得
        drug_effects = self.drug_system.describe_drug_effects()
        
        # 最新の説明を取得（または新しく生成）
        if not self.last_explanation:
            self.generate_behavior_explanation()
            
        # 状態の説明を組み合わせ
        state_explanation = {
            'neurotransmitter_levels': nt_state['neurotransmitter_levels'],
            'active_drugs': nt_state['active_drugs'],
            'behavioral_effects': nt_state['behavioral_effects'],
            'drug_effects': drug_effects,
            'behavior_explanation': self.last_explanation.get('summary', '説明がありません'),
            'confidence': self.last_explanation.get('confidence', 0.0)
        }
        
        return state_explanation
        
    def get_natural_language_explanation(self, query: str) -> str:
        """
        現在のシステム状態や行動に関する自然言語でのクエリに回答します。
        
        Args:
            query: 質問またはクエリ文字列
            
        Returns:
            質問に対する回答
        """
        try:
            # システム状態データを準備
            state = self.explain_current_state()
            
            # プロンプトの構築
            prompt = f"以下の神経科学的ヒューマノイドロボットの状態に関する質問に答えてください:\n\n"
            prompt += f"質問: {query}\n\n"
            
            # システム状態情報の追加
            prompt += "システム状態:\n"
            prompt += f"- 神経伝達物質レベル: {state['neurotransmitter_levels']}\n"
            prompt += f"- 活性化された薬物: {state['active_drugs']}\n"
            
            if state['active_drugs']:
                prompt += "- 薬物効果:\n"
                for drug, effect in state['drug_effects'].items():
                    prompt += f"  - {drug}: {effect}\n"
                    
            prompt += "\n行動への影響:\n"
            for nt, effect in state['behavioral_effects'].items():
                prompt += f"- {nt}: {effect}\n"
                
            if hasattr(self, 'last_motor_output'):
                prompt += f"\n現在の行動: {self.last_explanation.get('summary', '情報なし')}\n"
                prompt += f"行動の確信度: {state['confidence']:.2f}\n"
                
            # システムメッセージの設定
            system_message = """
            あなたは高度な神経科学とAIの専門家として、生物学的に妥当なヒューマノイドロボット制御システムについて説明します。
            質問に対して、神経伝達物質、受容体、薬物効果、皮質モデル、感覚運動統合に焦点を当てて、
            科学的に正確かつ分かりやすく回答してください。
            専門用語を適切に使用しつつ、一般の人にも理解できるよう努めてください。
            """
            
            # LLMから応答を生成
            response = self.llm_system.generate_response(prompt, system_message)
            return response
            
        except Exception as e:
            logger.error(f"説明の生成中にエラーが発生しました: {str(e)}")
            return f"説明の生成に失敗しました: {str(e)}"


def create_integrated_system(config_path: Optional[str] = None) -> IntegratedSystem:
    """
    統合システムのインスタンスを作成するファクトリ関数
    
    Args:
        config_path: 設定ファイルへのパス（オプション）
        
    Returns:
        初期化されたIntegratedSystemインスタンス
    """
    return IntegratedSystem(config_path) 