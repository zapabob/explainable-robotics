#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyPIにexplainable-roboticsパッケージを公開するためのスクリプト
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def clean_build_dirs():
    """ビルドディレクトリをクリーンアップ"""
    dirs_to_clean = [
        'build',
        'dist',
        'explainable_robotics.egg-info',
        '__pycache__',
        '.pytest_cache'
    ]
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"クリーンアップ: {dir_name}")
            shutil.rmtree(dir_name)
    
    # __pycache__ディレクトリを再帰的に検索して削除
    for pycache in Path('.').glob('**/__pycache__'):
        print(f"クリーンアップ: {pycache}")
        shutil.rmtree(pycache)

def build_package():
    """パッケージをビルド"""
    print("パッケージをビルドしています...")
    
    # ビルドコマンドの実行
    result = subprocess.run([sys.executable, '-m', 'build'], check=False)
    
    if result.returncode != 0:
        print("エラー: パッケージのビルドに失敗しました。")
        sys.exit(1)
    
    print("ビルド成功!")

def publish_to_pypi(test=True):
    """PyPIにパッケージを公開"""
    if test:
        repo_url = 'https://test.pypi.org/legacy/'
        print("TestPyPIにパッケージを公開しています...")
    else:
        repo_url = 'https://upload.pypi.org/legacy/'
        print("PyPIにパッケージを公開しています...")
    
    # Twineコマンドの実行
    result = subprocess.run([
        sys.executable, '-m', 'twine', 'upload', 
        '--repository-url', repo_url, 'dist/*'
    ], check=False)
    
    if result.returncode != 0:
        print(f"エラー: {'TestPyPI' if test else 'PyPI'}への公開に失敗しました。")
        sys.exit(1)
    
    print("公開成功!")
    
    if test:
        print("\nTestPyPIからインストールするには:")
        print("pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple explainable-robotics")
    else:
        print("\nPyPIからインストールするには:")
        print("pip install explainable-robotics")

def main():
    parser = argparse.ArgumentParser(description='explainable-roboticsパッケージのビルドと公開')
    parser.add_argument('--clean', action='store_true', help='ビルドディレクトリをクリーンアップ')
    parser.add_argument('--build', action='store_true', help='パッケージをビルド')
    parser.add_argument('--publish', action='store_true', help='PyPIに公開')
    parser.add_argument('--test', action='store_true', help='TestPyPIに公開 (デフォルト)')
    parser.add_argument('--production', action='store_true', help='本番PyPIに公開')
    
    args = parser.parse_args()
    
    # 引数がない場合はヘルプを表示
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    # 依存関係のチェック
    dependencies = ['build', 'twine']
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"エラー: 必要なパッケージがインストールされていません: {', '.join(missing)}")
        print("次のコマンドでインストールしてください:")
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)
    
    # アクションの実行
    if args.clean:
        clean_build_dirs()
    
    if args.build:
        build_package()
    
    if args.publish:
        # デフォルトではテスト環境に公開
        publish_to_pypi(not args.production)

if __name__ == '__main__':
    main() 