# explainable-robotics PyPI公開ガイド

## 概要

このドキュメントでは、`explainable-robotics`パッケージをPyPIに公開するための詳細な手順を説明します。プロジェクトはBioKAN、大脳皮質モデル、LangchainのマルチモーダルLLM、Genesis（`import genesis as gs`）を統合した総合的な脳機能システムです。

## 前提条件

1. Python 3.12以上がインストールされていること
2. PyPIアカウントが作成されていること
3. 必要なパッケージがインストールされていること
   ```bash
   pip install build twine
   ```

## 公開手順

### 1. 準備

1. `~/.pypirc`ファイルを作成（または更新）して認証情報を設定
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

2. プロジェクトのバージョン確認
   - `explainable_robotics/__init__.py`の`__version__`の値を確認
   - 必要に応じてバージョンを更新

### 2. パッケージのビルドとテスト公開

1. 付属の公開スクリプトを使用する場合:
   ```bash
   python publish.py --clean --build --publish --test
   ```

2. 手動でビルドとテスト公開を行う場合:
   ```bash
   # ビルドディレクトリのクリーンアップ
   rmdir /s /q build dist explainable_robotics.egg-info

   # パッケージのビルド
   python -m build

   # テストPyPIにアップロード
   python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   ```

3. テストインストールの確認
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple explainable-robotics
   ```

4. テストパッケージの動作確認
   ```python
   import explainable_robotics
   print(explainable_robotics.__version__)

   # Genesis as gsのインポート確認
   from explainable_robotics.genesis import gs
   print(dir(gs))

   # BioKANとの統合確認
   from explainable_robotics.cortical import BioKAN
   ```

### 3. 本番公開

問題がなければ本番環境に公開します。

1. 付属の公開スクリプトを使用する場合:
   ```bash
   python publish.py --clean --build --publish --production
   ```

2. 手動で本番公開を行う場合:
   ```bash
   # クリーンビルド
   rmdir /s /q build dist explainable_robotics.egg-info
   python -m build

   # PyPIへのアップロード
   python -m twine upload dist/*
   ```

3. インストールと確認
   ```bash
   pip install explainable-robotics
   ```

## 依存関係の注意点

このパッケージは以下の主要な依存関係があります:

- `genesis-world`: Genesisシミュレーション環境（`import genesis as gs`でインポート）
- `biokan`: 生物学的知識グラフデータベース
- `langchain`と関連モジュール: マルチモーダルLLM統合
- `torch`: 神経ネットワークモデル

オプションの依存関係はextrasで定義されています:
- 開発ツール: `pip install explainable-robotics[dev]`
- GPU高速化: `pip install explainable-robotics[gpu]`
- すべての機能: `pip install explainable-robotics[all]`

## トラブルシューティング

1. アップロード時の認証エラー
   - コマンドプロンプトで直接ユーザー名とパスワードを入力
   - `~/.pypirc`の権限が適切か確認（Linuxの場合は600）

2. 依存関係の解決エラー
   - `setup.py`のバージョン指定が適切か確認
   - 互換性のない依存関係の組み合わせがないか確認

3. ビルドエラー
   - `MANIFEST.in`に必要なファイルがすべて含まれているか確認
   - 各サブパッケージに`__init__.py`が存在するか確認
   - Python 3.12との互換性を確認

## リリース後のメンテナンス

1. 定期的なバージョンアップデート
   - 新機能追加: マイナーバージョンを上げる (0.1.0 → 0.2.0)
   - バグ修正: パッチバージョンを上げる (0.1.0 → 0.1.1)

2. PyPI上での情報更新
   - プロジェクトの説明、ホームページなどの確認
   - リリースノートの追加

3. フィードバックと貢献
   - GitHub IssueやPull Requestの管理
   - バグ報告や機能リクエストの処理

## 連絡先

問題や質問がある場合は、プロジェクトの GitHub レポジトリを通じてお問い合わせください。 