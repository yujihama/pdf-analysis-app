import logging
import sys
import os
from datetime import datetime

# 文字エンコーディングを明示的に設定
if sys.stdout.encoding != 'utf-8':
    print(f"現在の標準出力エンコーディング: {sys.stdout.encoding}")
    print("UTF-8に変更します")
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    print(f"変更後の標準出力エンコーディング: {sys.stdout.encoding}")

# ログディレクトリを作成
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# ログファイル名の設定
log_filename = f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(log_dir, log_filename)

print(f"ログファイルパス: {log_filepath}")
print(f"カレントディレクトリ: {os.getcwd()}")
print(f"実行ファイル: {__file__}")
print(f"コマンドライン引数: {sys.argv}")

# ロギングの設定
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8', mode='w')

# フォーマッタの設定
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 既存のロガーをリセット
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ロガーの設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# テストメッセージをログに出力
logger.info("これはテストログメッセージです")
logger.info(f"Pythonのバージョン: {sys.version}")
logger.info(f"実行環境: {os.environ.get('PYTHONPATH', '未設定')}")
logger.info(f"標準出力エンコーディング: {sys.stdout.encoding}")

print("ログテスト完了。logs ディレクトリを確認してください。") 