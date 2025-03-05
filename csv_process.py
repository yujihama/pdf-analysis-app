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
            temperature=temperature,
            response_format={"type": "json_object"}
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
            temperature=temperature,
            response_format={"type": "json_object"}
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

def process_all_chunks(csv_data, prompt_template, max_tokens, filter_key=None, filter_value=None):
    """
    CSVの全チャンクを処理する
    
    Args:
        csv_data: CSVデータのDataFrame（1カラムのみ）
        prompt_template: プロンプトテンプレート
        max_tokens: 最大トークン数
        filter_key: フィルタリングするJSONのキー名
        filter_value: フィルタリングする値
    
    Returns:
        各チャンクの回答のリスト、フィルタリング後の回答のリスト、および入力データと回答を合わせたDataFrame
    """
    results = []
    filtered_results = []
    filtered_indices = []
    progress_bar = st.progress(0)
    
    # カラム名を取得（最初の1つのみ使用）
    if len(csv_data.columns) >= 1:
        column_name = csv_data.columns[0]
    else:
        st.error("CSVファイルには少なくとも1つのカラムが必要です。")
        return [], [], None
    
    # 処理の進捗状況を表示
    for i, row in enumerate(csv_data[column_name]):
        progress_percent = (i + 1) / len(csv_data)
        progress_bar.progress(progress_percent)
        
        with st.spinner(f"チャンク {i+1}/{len(csv_data)} を処理中..."):
            # チャンクを処理して回答を取得
            response = process_csv_chunk(row, prompt_template, max_tokens)
            results.append(response)
            
            # フィルタリング条件が指定されている場合、JSONレスポンスを解析してフィルタリング
            if filter_key and filter_value:
                try:
                    # JSONとして解析
                    response_json = json.loads(response)
                    
                    # 指定されたキーが存在し、値に指定文字列が含まれる場合にフィルタリング結果に追加（大文字小文字を区別しない）
                    if filter_key in response_json and str(filter_value).lower() in str(response_json[filter_key]).lower():
                        filtered_results.append(response)
                        filtered_indices.append(i)
                        logger.info(f"チャンク {i+1} はフィルター条件に一致しました: {filter_key}に「{filter_value}」が含まれています")
                    else:
                        logger.info(f"チャンク {i+1} はフィルター条件に一致しませんでした")
                except json.JSONDecodeError:
                    logger.warning(f"チャンク {i+1} の応答はJSON形式ではありません")
            else:
                # フィルタリング条件がない場合は全ての結果を対象とする
                filtered_results.append(response)
                filtered_indices.append(i)
    
    # 結果のDataFrameを作成
    result_df = pd.DataFrame({
        'input_text': csv_data[column_name],
        'response': results,
        'filtered': [i in filtered_indices for i in range(len(results))]
    })
    
    return results, filtered_results, result_df

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
    if 'filtered_responses' not in st.session_state:
        st.session_state.filtered_responses = []
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'integrated_response' not in st.session_state:
        st.session_state.integrated_response = None
    if 'json_keys' not in st.session_state:
        st.session_state.json_keys = []
    if 'just_filtered' not in st.session_state:
        st.session_state.just_filtered = False
    if 'filter_applied' not in st.session_state:
        st.session_state.filter_applied = False
    
    # 再フィルタリング用関数
    def apply_filter():
        if not st.session_state.processed:
            return
            
        try:
            # 現在のフィルターキーと値を取得
            current_filter_key = st.session_state.current_filter_key
            current_filter_value = st.session_state.current_filter_value
            
            if not current_filter_key or not current_filter_value:
                st.warning("フィルタリングするにはキーと値の両方が必要です")
                return
                
            filtered_results = []
            filtered_indices = []
            
            # フィルタリング条件に基づいて結果を再フィルタリング
            for i, response in enumerate(st.session_state.responses):
                try:
                    response_json = json.loads(response)
                    # 完全一致ではなく、部分一致に変更（含まれていれば該当と判定）
                    if current_filter_key in response_json and str(current_filter_value).lower() in str(response_json[current_filter_key]).lower():
                        filtered_results.append(response)
                        filtered_indices.append(i)
                except json.JSONDecodeError:
                    continue
            
            # フィルタリング結果を更新
            st.session_state.filtered_responses = filtered_results
            
            # DataFrameのフィルタリング列を更新
            if st.session_state.result_df is not None:
                st.session_state.result_df['filtered'] = [i in filtered_indices for i in range(len(st.session_state.responses))]
            
            # フィルタリング状態を更新
            st.session_state.filter_applied = True
            st.session_state.just_filtered = True
            
            logger.info(f"フィルタリングが適用されました: {current_filter_key}に「{current_filter_value}」が含まれる")
        except Exception as e:
            st.error(f"フィルタリング処理でエラーが発生しました: {str(e)}")
            logger.error(f"フィルタリングエラー: {str(e)}", exc_info=True)
    
    # サイドバー設定
    st.sidebar.header("設定")
    max_tokens = st.sidebar.number_input("max_tokens", min_value=100, max_value=9000, value=5000, step=100)
    
    # フィルタリング設定
    st.sidebar.header("フィルタリング設定")
    use_filter = st.sidebar.checkbox("JSONのキーでフィルタリングする", value=st.session_state.filter_applied)
    
    if use_filter:
        # 処理済みでJSONキーが抽出されている場合は選択式で表示
        if st.session_state.processed and st.session_state.json_keys:
            # フィルターキーのセレクトボックス
            filter_key = st.sidebar.selectbox(
                "フィルタリングするJSONキー", 
                options=st.session_state.json_keys,
                key="current_filter_key"
            )
            
            if filter_key:
                # 参考情報を表示するための値を収集
                unique_values = []
                try:
                    for response in st.session_state.responses:
                        try:
                            response_json = json.loads(response)
                            if filter_key in response_json:
                                value = str(response_json[filter_key])
                                if value not in unique_values:
                                    unique_values.append(value)
                        except json.JSONDecodeError:
                            continue
                except:
                    pass
                
                # 値の入力フィールド
                filter_value = st.sidebar.text_input(
                    "フィルタリングする値", 
                    key="current_filter_value"
                )
                
                # 参考として存在する値を表示
                if unique_values:
                    values_str = ", ".join(unique_values)
                    st.sidebar.info(f"参考: このキーの値の例: {values_str}")
                
                # フィルター適用ボタン
                if st.sidebar.button("フィルターを適用"):
                    apply_filter()
            else:
                st.sidebar.text_input("フィルタリングする値", key="current_filter_value")
        else:
            # 処理前やJSONキーがない場合はテキスト入力
            st.sidebar.info("CSVを処理すると、JSONキーが選択式で表示されます。")
            st.sidebar.text_input("フィルタリングするJSONキー", key="current_filter_key")
            st.sidebar.text_input("フィルタリングする値", key="current_filter_value")
            
        st.sidebar.info("指定したキーの値が一致するレスポンスのみが統合の対象になります。両方の項目を入力してください。")
        
        # フィルターのリセットボタン
        if st.session_state.filter_applied:
            if st.sidebar.button("フィルターをリセット"):
                st.session_state.filtered_responses = st.session_state.responses.copy()
                if st.session_state.result_df is not None:
                    st.session_state.result_df['filtered'] = [True] * len(st.session_state.responses)
                st.session_state.filter_applied = False
                st.session_state.just_filtered = False
    
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
                "以下の引用文書の情報を分析して要約してください。重要なポイントを箇条書きで5つ挙げてください。JSONフォーマットで回答してください。",
                height=150
            )
            
            prompt2 = st.text_area(
                "回答統合用プロンプト（オプション）", 
                "以下の回答を統合して、共通する重要なポイントと特徴的な違いを分析してください。全体的な傾向と特筆すべき点を簡潔にまとめてください。",
                height=150
            )
            
            # 処理実行ボタン
            if st.button("処理開始"):
                # フィルタリング設定が不完全な場合の警告
                if use_filter and (not st.session_state.get("current_filter_key") or not st.session_state.get("current_filter_value")):
                    st.warning("フィルタリングを有効にする場合は、キーと値の両方を入力してください。")
                
                with st.spinner("CSVデータを処理中..."):
                    # フィルタリング条件を設定
                    filter_key_param = st.session_state.get("current_filter_key") if use_filter else None
                    filter_value_param = st.session_state.get("current_filter_value") if use_filter else None
                    
                    # 各チャンクを処理
                    responses, filtered_responses, result_df = process_all_chunks(
                        df, prompt1, max_tokens, filter_key_param, filter_value_param
                    )
                    
                    # JSONキーを抽出
                    json_keys = set()
                    for response in responses:
                        try:
                            response_json = json.loads(response)
                            for key in response_json.keys():
                                json_keys.add(key)
                        except json.JSONDecodeError:
                            continue
                    
                    # セッション状態に結果を保存
                    st.session_state.processed = True
                    st.session_state.responses = responses
                    st.session_state.filtered_responses = filtered_responses
                    st.session_state.result_df = result_df
                    st.session_state.json_keys = list(json_keys)
                    
                    # 結果の表示
                    st.success("処理が完了しました！")

            # 処理済みの場合は結果を表示
            if st.session_state.processed and st.session_state.result_df is not None:
                st.subheader("処理結果")
                
                # 再フィルタリング直後のメッセージを表示
                if st.session_state.just_filtered:
                    st.success("フィルタリングが適用されました！")
                    # メッセージは一度だけ表示するためにフラグをリセット
                    st.session_state.just_filtered = False
                
                # フィルタリング状態を表示
                if st.session_state.filter_applied and use_filter:
                    total_chunks = len(st.session_state.responses)
                    filtered_chunks = len(st.session_state.filtered_responses)
                    current_filter_key = st.session_state.get("current_filter_key", "")
                    current_filter_value = st.session_state.get("current_filter_value", "")
                    st.info(f"全 {total_chunks} チャンク中、{filtered_chunks} チャンクがフィルター条件 「{current_filter_key}に「{current_filter_value}」が含まれる」 に一致しました。")
                
                # 結果のDataFrameを表示
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
                            # フィルタリングされた結果を使用して統合
                            responses_to_integrate = st.session_state.filtered_responses if use_filter else st.session_state.responses
                            
                            if not responses_to_integrate:
                                st.warning("フィルター条件に一致する回答がありません。フィルタリング条件を見直すか、フィルタリングを無効にしてください。")
                            else:
                                integrated_response = integrate_responses(responses_to_integrate, prompt2)
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