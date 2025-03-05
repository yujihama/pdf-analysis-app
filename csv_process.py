import streamlit as st
import openai
import pandas as pd
import io
import logging
import sys
from datetime import datetime
import os
import csv
import json
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv()

# ロギングの設定
# ログディレクトリを作成
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log_filename = f'csv_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
log_filepath = os.path.join(log_dir, log_filename)

# フォーマッタの設定
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# ハンドラの設定
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_filepath, encoding='utf-8', mode='w')
file_handler.setFormatter(formatter)

# 既存のロガーをリセット
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ロガーの設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info(f"ログファイルを作成しました: {log_filepath}")

######################
# OpenAI APIキー設定 #
######################
# Azure OpenAI APIの設定を確認
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

# Azure OpenAI APIが設定されている場合はAzure OpenAI APIを使用
if azure_api_key and azure_endpoint and azure_deployment:
    logger.info("Azure OpenAI APIを使用します")
    openai.api_type = "azure"
    openai.api_key = azure_api_key
    openai.api_base = azure_endpoint
    openai.api_version = azure_api_version
    default_model = azure_deployment
else:
    # 環境変数からOpenAI APIキーを取得
    logger.info("OpenAI APIを使用します")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    default_model = "gpt-4o"  # デフォルトモデル
    
    if not openai.api_key:
        st.error("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")
        st.stop()

##############################
# LLM呼び出しのユーティリティ #
##############################
def call_gpt(prompt, max_tokens=1500, temperature=0.0, model=None):
    """
    LLMを呼び出すユーティリティ関数
    Args:
        prompt: プロンプトテキスト
        max_tokens: 最大トークン数
        temperature: 温度パラメータ
        model: 使用するモデル（Noneの場合はデフォルトモデルを使用）
    """
    # モデルが指定されていない場合はデフォルトモデルを使用
    if model is None:
        model = default_model
        
    messages = [
        {"role": "system", "content": "あなたは与えられた情報に基づいて正確に回答するアシスタントです。"}
    ]

    logger.info("==========call_gpt start==========")
    logger.info(f"使用モデル: {model}")
    
    messages.append({"role": "user", "content": prompt})

    # APIの種類に応じてAPIを呼び出す
    if openai.api_type == "azure":
        # Azure OpenAI APIの場合
        # 明示的にAzure OpenAIクライアントを初期化
        client = openai.AzureOpenAI(
            api_key=openai.api_key,
            api_version=openai.api_version,
            azure_endpoint=openai.api_base
        )
        response = client.chat.completions.create(
            deployment_id=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.info("==========call_gpt end==========")
        return response.choices[0].message.content
    else:
        # 通常のOpenAI APIの場合
        # 明示的にクライアントを初期化
        client = openai.OpenAI(
            api_key=openai.api_key
        )
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.info("==========call_gpt end==========")
        return response.choices[0].message.content

def process_csv_chunk(chunk_data, prompt_template, max_tokens):
    """
    CSVの1チャンクを処理し、GPTによる回答を生成する
    
    Args:
        chunk_data: CSVの1行のテキストデータ（単一カラム）
        prompt_template: プロンプトテンプレート
    
    Returns:
        生成された回答
    """
    # チャンクデータを文字列として使用（すでに文字列形式）
    chunk_text = chunk_data
    
    # プロンプトにチャンクデータを挿入
    full_prompt = f"{prompt_template}\n\n引用文書:\n```\n{chunk_text}\n```"
    
    # GPTを呼び出して回答を生成
    response = call_gpt(full_prompt, max_tokens=max_tokens, model="gpt-4o")
    
    return response

def process_all_chunks(csv_data, prompt_template, max_tokens):
    """
    CSVの全チャンクを処理する
    
    Args:
        csv_data: CSVデータのDataFrame（1カラムのみ）
        prompt_template: プロンプトテンプレート
    
    Returns:
        各チャンクの回答のリスト、および入力データと回答を合わせたDataFrame
    """
    results = []
    progress_bar = st.progress(0)
    
    # カラム名を取得（最初の1つのみ使用）
    if len(csv_data.columns) >= 1:
        column_name = csv_data.columns[0]
    else:
        st.error("CSVファイルには少なくとも1つのカラムが必要です。")
        return [], None
    
    # 処理の進捗状況を表示
    for i, row in enumerate(csv_data[column_name]):
        progress_percent = (i + 1) / len(csv_data)
        progress_bar.progress(progress_percent)
        
        with st.spinner(f"チャンク {i+1}/{len(csv_data)} を処理中..."):
            # チャンクを処理して回答を取得
            response = process_csv_chunk(row, prompt_template, max_tokens)
            results.append(response)
    
    # 結果のDataFrameを作成
    result_df = pd.DataFrame({
        'input_text': csv_data[column_name],
        'response': results
    })
    
    return results, result_df

def integrate_responses(responses, integration_prompt):
    """
    全回答を統合するプロンプトを実行
    
    Args:
        responses: 各チャンクの回答のリスト
        integration_prompt: 統合用のプロンプト
    
    Returns:
        統合された回答
    """
    # 全回答を1つの文字列にまとめる
    all_responses_text = ""
    for i, response in enumerate(responses):
        all_responses_text += f"\n\n===== 回答 {i+1} =====\n{response}"
    
    # 統合プロンプトを作成
    full_prompt = f"{integration_prompt}\n\n引用文書:\n```\n{all_responses_text}\n```"
    
    # GPTを呼び出して統合回答を生成
    integrated_response = call_gpt(full_prompt, max_tokens=2000, model="gpt-4o")
    
    return integrated_response

def main():
    st.title("Map Reduce")    
    # セッション状態の初期化
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'responses' not in st.session_state:
        st.session_state.responses = []
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'integrated_response' not in st.session_state:
        st.session_state.integrated_response = None
    
    # サイドバー設定
    st.sidebar.header("設定")
    max_tokens = st.sidebar.number_input("max_tokens", min_value=100, max_value=9000, value=5000, step=100)
    
    # CSVファイルアップロード
    uploaded_file = st.file_uploader("CSVファイルをアップロード（1カラムのみ）", type=['csv'])
    
    if uploaded_file is not None:
        # CSVファイルの読み込み
        try:
            df = pd.read_csv(uploaded_file)
            
            # 1カラム目のみを使用する
            if len(df.columns) >= 1:
                column_name = df.columns[0]
                st.write(f"CSVファイルを読み込みました。行数: {len(df)}")
                st.write("CSVプレビュー (1カラム目のみ使用):")
                
                # プレビューの表示（1カラム目のみ）
                preview_df = pd.DataFrame({column_name: df[column_name]})
                st.dataframe(preview_df.head())
            else:
                st.error("CSVファイルには少なくとも1つのカラムが必要です。")
                st.stop()
            
            # プロンプト入力欄
            st.subheader("プロンプト設定")
            prompt1 = st.text_area(
                "各チャンク処理用プロンプト", 
                "以下の引用文書の情報を分析して要約してください。重要なポイントを箇条書きで5つ挙げてください。",
                height=150
            )
            
            prompt2 = st.text_area(
                "回答統合用プロンプト（オプション）", 
                "以下の回答を統合して、共通する重要なポイントと特徴的な違いを分析してください。全体的な傾向と特筆すべき点を簡潔にまとめてください。",
                height=150
            )
            
            # 処理実行ボタン
            if st.button("処理開始"):
                with st.spinner("CSVデータを処理中..."):
                    # 各チャンクを処理
                    responses, result_df = process_all_chunks(df, prompt1, max_tokens)
                    
                    # セッション状態に結果を保存
                    st.session_state.processed = True
                    st.session_state.responses = responses
                    st.session_state.result_df = result_df
                    
                    # 結果の表示
                    st.success("処理が完了しました！")

            # 処理済みの場合は結果を表示
            if st.session_state.processed and st.session_state.result_df is not None:
                st.subheader("処理結果")
                st.dataframe(st.session_state.result_df)
                
                # 結果のCSVダウンロードボタン
                csv_buffer = io.StringIO()
                st.session_state.result_df.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                st.download_button(
                    label="処理結果をCSVでダウンロード",
                    data=csv_str,
                    file_name="processed_results.csv",
                    mime="text/csv"
                )
            
            # 処理済みで、回答統合プロンプトが入力されている場合は統合ボタンを表示
            if st.session_state.processed and prompt2.strip():
                if st.button("回答を統合"):
                    with st.spinner("回答を統合中..."):
                        integrated_response = integrate_responses(st.session_state.responses, prompt2)
                        st.session_state.integrated_response = integrated_response
                
            # 統合回答がある場合は表示
            if st.session_state.integrated_response:
                st.subheader("統合回答")
                st.write(st.session_state.integrated_response)
                
                # 統合回答のダウンロードボタン
                st.download_button(
                    label="統合回答をテキストファイルでダウンロード",
                    data=st.session_state.integrated_response,
                    file_name="integrated_response.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"エラー詳細: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 