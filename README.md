# Explainable Robotics

神経科学的に妥当なヒューマノイドロボット制御のための説明可能AIフレームワーク

[![Python Tests](https://github.com/yourusername/explainable-robotics/actions/workflows/python-tests.yml/badge.svg)](https://github.com/yourusername/explainable-robotics/actions/workflows/python-tests.yml)
[![PyPI version](https://badge.fury.io/py/explainable-robotics.svg)](https://badge.fury.io/py/explainable-robotics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要

Explainable Roboticsは、神経科学的に妥当なモデルを用いてヒューマノイドロボットの行動を制御し、その意思決定過程を説明可能にするフレームワークです。本プロジェクトは以下の3つの主要コンポーネントから構成されています：

1. **BioKAN** - 大脳皮質の6層構造を模倣したKolmogorov-Arnold Network（KAN）に基づく神経回路モデル
2. **MultiLLMAgent** - OpenAI、Claude、Geminiなどの大規模言語モデルを使用した意思決定エージェント
3. **Genesis** - リアルタイムの物理シミュレーションと視覚化エンジン

## インストール方法

```bash
pip install explainable-robotics
```

必要に応じて特定のバージョンを指定することも可能です：

```bash
pip install explainable-robotics==0.1.0
```

## 使用例

### 基本的な使用方法

```python
from explainable_robotics import create_integrated_system

# 統合システムの作成
system = create_integrated_system(
    robot_name="MyExplainableRobot",
    biokan_config={"layers": 6, "neurons_per_layer": 100},
    llm_config={"provider": "openai", "model": "gpt-4"}
)

# システムの起動
system.start()

# 目標の設定
system.set_goal("物体を把握して移動する")

# 実行（例：30秒間）
import time
time.sleep(30)

# システムの停止
system.stop()
```

### 対話モードでの使用

```bash
python -m explainable_robotics.examples.kan_llm_genesis_integration --interactive
```

### 神経伝達物質レベルの調整

```python
# 神経伝達物質レベルの調整
system.adjust_neurotransmitter_levels({
    "dopamine": 0.7,  # 意欲と報酬
    "serotonin": 0.5,  # 感情バランス
    "noradrenaline": 0.6,  # 覚醒と注意
    "acetylcholine": 0.5,  # 認知と記憶
    "glutamate": 0.5,  # 興奮性
    "gaba": 0.5  # 抑制性
})
```

## 主な機能

- **三値入力処理** - 抑制（-1）、中立（0）、興奮（1）の3つの値を処理する神経学的に妥当なモデル
- **複数LLMサポート** - OpenAI、Claude、Geminiの複数のLLMプロバイダに対応
- **フォールバックメカニズム** - 利用可能なLLMに自動的に切り替えるフォールバック機能
- **物理シミュレーション** - リアルタイムの物理世界とのインタラクション
- **説明可能な決定プロセス** - 意思決定過程の透明性と説明機能
- **適応型行動** - 環境や目標に応じて行動を適応的に調整

## ドキュメント

詳細なドキュメントは[こちら](https://yourusername.github.io/explainable-robotics)を参照してください。

## 貢献方法

プロジェクトへの貢献に興味をお持ちの方は、[CONTRIBUTING.md](CONTRIBUTING.md)をご覧ください。

## ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルをご参照ください。