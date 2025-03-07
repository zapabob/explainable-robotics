# PyPI公開手順

このドキュメントでは、explainable-roboticsパッケージをPyPIに公開する手順を説明します。

## 準備

1. 必要なパッケージをインストール:

```bash
pip install build twine
```

2. PyPIアカウントの準備:
   - [PyPI](https://pypi.org/)と[TestPyPI](https://test.pypi.org/)にアカウントを作成
   - `~/.pypirc`ファイルに認証情報を設定（オプション）:

```
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = your_username
password = your_password

[testpypi]
repository = https://test.pypi.org/legacy/
username = your_username
password = your_password
```

## パッケージの公開

### 公開スクリプトの使用

このリポジトリには公開用のスクリプト`publish.py`が含まれています。

1. クリーンアップ、ビルド、TestPyPIへの公開:

```bash
python publish.py --clean --build --publish --test
```

2. 本番PyPIへの公開:

```bash
python publish.py --clean --build --publish --production
```

### 手動での公開

手動で公開する場合は、以下の手順に従います：

1. ビルドディレクトリのクリーンアップ:

```bash
rm -rf build/ dist/ *.egg-info
```

2. パッケージのビルド:

```bash
python -m build
```

3. TestPyPIへのアップロード:

```bash
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

4. TestPyPIからのインストールテスト:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple explainable-robotics
```

5. 本番PyPIへのアップロード:

```bash
python -m twine upload dist/*
```

## パッケージ構造

explainable-roboticsのパッケージ構造:

```
explainable-robotics/
├── explainable_robotics/
│   ├── __init__.py
│   ├── main.py
│   ├── cli.py
│   ├── core/
│   ├── cortical/
│   ├── demos/
│   ├── genesis/
│   ├── llm/
│   ├── visualization/
│   └── utils/
├── MANIFEST.in
├── LICENSE
├── pyproject.toml
├── README.md
├── setup.py
└── publish.py
```

## 依存関係

主な依存関係:

- Python 3.12以上
- PyTorch
- LangChain
- Genesis-world
- BioKAN

オプションの依存関係は`setup.py`に定義されています。

## バージョニング

セマンティックバージョニングを使用しています:

- メジャーバージョン: 互換性のない変更
- マイナーバージョン: 後方互換性のある新機能
- パッチバージョン: バグ修正

バージョンは`explainable_robotics/__init__.py`の`__version__`変数で定義されています。

## トラブルシューティング

1. アップロード時の認証問題:
   - `~/.pypirc`ファイルが正しく設定されているか確認
   - または、コマンドラインで認証情報を入力

2. 依存関係の問題:
   - `setup.py`の依存関係が正しいか確認
   - バージョン指定が適切か確認

3. パッケージの構造:
   - `MANIFEST.in`が必要なファイルを含んでいるか確認
   - `__init__.py`ファイルがすべてのディレクトリに存在するか確認 