# ExplainableRobotics

神経科学的に妥当なヒューマノイドロボット制御のための説明可能AIフレームワーク

## 概要

ExplainableRoboticsは、生物学的に妥当な神経ネットワークモデルを使用してヒューマノイドロボットを制御するためのフレームワークです。このライブラリは以下の特徴を持っています：

- 大脳皮質の6層構造を模倣した神経ネットワークモデル
- BioKANとの統合による生物学的妥当性の向上
- Genesisロボティクスシミュレーションとの連携
- アセチルコリン、ドーパミン、セロトニン、ノルアドレナリン、グルタミン酸、GABAなどの神経伝達物質レベルの調整
- さまざまな中枢神経系薬物の効果のシミュレーション
- 自然言語でモデルの意思決定を説明する機能
- マルチモーダルLLMとの統合による拡張された説明生成

## インストール

### 前提条件

- Python 3.12以上
- PyTorch 1.8以上
- Genesis-world

### インストール手順

```bash
# リポジトリのクローン
git clone https://github.com/zapabob/explainable_robotics.git
cd explainable_robotics

# 依存関係のインストール
pip install -e .
```

## 基本的な使い方

### コマンドラインインターフェース（CLI）

ExplainableRoboticsは、使いやすいコマンドラインインターフェースを提供しています。

```bash
# デフォルト設定ファイルの作成
python -m explainable_robotics.main --create-config

# システムの開始
python -m explainable_robotics.main --config config/default_config.json
```

CLI内で利用可能なコマンド：

- `start` - システムを開始
- `stop` - システムを停止
- `state` - システム状態を表示
- `nt` - 神経伝達物質レベルを表示
- `set_nt <type> <level>` - 神経伝達物質レベルを設定 (0.0-1.0)
- `drug <name> [dose]` - 薬物を適用
- `explain` - 現在の行動の説明を生成
- `ask <question>` - システムに質問
- `gesture <name>` - ジェスチャーを実行
- `graph` - 神経伝達物質レベルをグラフ表示
- `config` - 現在の設定を表示
- `exit` / `quit` - 終了

### Pythonプログラミング例

```python
from explainable_robotics.core.integrated_system import create_integrated_system

# システムの初期化
system = create_integrated_system('config/default_config.json')

# 知識ベースの初期化
system.initialize_knowledge_base()

# システムの開始
system.start()

# 神経伝達物質レベルの調整
system.adjust_neurotransmitter('dopamine', 0.8)
system.adjust_neurotransmitter('glutamate', 0.7)

# 薬物の適用
system.apply_drug('methylphenidate', 0.5)  # リタリン

# 行動の説明を生成
explanation = system.generate_behavior_explanation()
print(explanation['summary'])

# システムに質問
response = system.get_natural_language_explanation(
    "グルタミン酸とGABAのバランスはどのように行動に影響しますか？"
)
print(response)

# ジェスチャーの実行
system.execute_gesture('wave')

# システムの停止
system.stop()
```

## 設定ファイルのカスタマイズ

設定ファイルは以下のセクションに分かれています：

### システム設定

```json
"system": {
  "data_dir": "./data",
  "knowledge_dir": "./data/knowledge",
  "log_dir": "./logs",
  "explanation_dir": "./data/explanations"
}
```

### 神経伝達物質設定

```json
"neurotransmitters": {
  "default_levels": {
    "acetylcholine": 0.5,
    "dopamine": 0.5,
    "serotonin": 0.5,
    "noradrenaline": 0.5,
    "glutamate": 0.5,
    "gaba": 0.5
  },
  "receptor_sensitivities": {
    "glutamate": {
      "nmda": 1.0,
      "ampa": 1.0,
      "kainate": 1.0,
      "mglur": 1.0
    },
    "gaba": {
      "gaba_a": 1.0,
      "gaba_b": 1.0
    }
  }
}
```

### 皮質モデル設定

```json
"cortical_model": {
  "layers": ["layer1", "layer2_3", "layer4", "layer5", "layer6"],
  "input_dim": 100,
  "output_dim": 22,
  "activation_functions": {
    "layer1": "relu",
    "layer2_3": "tanh",
    "layer4": "relu",
    "layer5": "tanh",
    "layer6": "tanh"
  }
}
```

### ロボットインターフェース設定

```json
"robot_interface": {
  "simulation_mode": true,
  "connection": {
    "type": "usb",
    "port": "COM3",
    "baudrate": 115200
  },
  "sensors": {
    "camera": true,
    "imu": true,
    "joint_sensors": true,
    "force_sensors": true
  }
}
```

### LLM設定

```json
"llm": {
  "default_provider": "openai",
  "openai": {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "anthropic": {
    "model": "claude-3-opus-20240229",
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

## 実験と可視化

### 神経伝達物質レベルの可視化

システムは、神経伝達物質レベルをグラフとして可視化する機能を提供しています。

```python
# CLIから
explainable_robotics> graph

# Pythonから
from explainable_robotics.demo import visualize_neurotransmitter_levels

state = system.explain_current_state()
visualize_neurotransmitter_levels(state['neurotransmitter_levels'], "現在の神経伝達物質レベル")
```

### デモの実行

様々な神経科学的実験を実行するデモスクリプトも用意されています。

```bash
python -m explainable_robotics.demo
```

デモでは以下の実験が行われます：

1. ベースラインのシステム状態
2. ドーパミンレベルの調整による行動変化
3. グルタミン酸とGABAのバランス調整
4. さまざまな中枢神経系薬物の効果

## API リファレンス

主要なクラスとメソッド：

### IntegratedSystem

```python
from explainable_robotics.core.integrated_system import IntegratedSystem

# メソッド
system.initialize_knowledge_base()  # 知識ベースの初期化
system.apply_drug(drug_name, dose)  # 薬物の適用
system.adjust_neurotransmitter(transmitter_type, level)  # 神経伝達物質レベルの調整
system.adjust_receptor_sensitivity(transmitter_type, receptor_changes)  # 受容体感受性の調整
system.process_sensor_data(sensor_data)  # センサーデータの処理
system.generate_behavior_explanation()  # 行動の説明を生成
system.get_natural_language_explanation(query)  # 自然言語での質問への回答
system.execute_gesture(gesture_name)  # ジェスチャーの実行
system.start()  # システムの開始
system.stop()  # システムの停止
```

### GenesisRobotInterface

```python
from explainable_robotics.genesis.robot_interface import GenesisRobotInterface

# メソッド
robot.read_sensors()  # センサーデータの読み取り
robot.send_motor_commands(motor_commands)  # モーターコマンドの送信
robot.convert_cortical_output_to_motor_commands(cortical_output)  # 皮質出力からモーターコマンドへの変換
robot.execute_gesture(gesture_name)  # ジェスチャーの実行
robot.calibrate()  # ロボットのキャリブレーション
robot.shutdown()  # ロボットの安全なシャットダウン
```

### MultimodalLLMSystem

```python
from explainable_robotics.llm.langchain_integration import MultimodalLLMSystem

# メソッド
llm.create_knowledge_base(documents_dir)  # 知識ベースの作成
llm.search_knowledge_base(query)  # 知識ベースの検索
llm.generate_response(prompt, system_message)  # LLMからの応答生成
llm.explain_decision(sensor_data, motor_output, layer_activations, nt_levels)  # 意思決定の説明
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細はLICENSEファイルを参照してください。

## 引用

このプロジェクトを学術研究で使用する場合は、以下の方法で引用してください：

```
@software{explainable_robotics,
  author = {Ryo Minegishi},
  title = {ExplainableRobotics: A Biologically Plausible Framework for Humanoid Robot Control},
  year = {2025},
  url = {https://github.com/zapabob/explainable_robotics}
}
``` 
