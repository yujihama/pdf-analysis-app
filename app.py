import streamlit as st
import openai
import PyPDF2
import json
import io
import base64
import fitz  # PyMuPDF
from PIL import Image
import logging
import sys
from datetime import datetime
import os
import shutil
import uuid
import cv2
import numpy as np
from dotenv import load_dotenv
import traceback
import os.path
# 画像分析モジュールのインポート
# import image_analysis

# .envファイルの読み込み
load_dotenv()

# ロギングの設定
# 文字エンコーディングを明示的に設定
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass  # Streamlit環境では例外が発生する可能性があるため

# ログディレクトリを作成
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log_filename = f'pdf_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# バッファリングを無効化
for handler in logger.handlers:
    if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
        handler.flush()

logger.info(f"ログファイルを作成しました: {log_filepath}")
logger.info(f"標準出力エンコーディング: {sys.stdout.encoding}")

# 環境変数の情報をログに記録
logger.info(f"環境設定: ENABLE_FILE_LOGGING={os.getenv('ENABLE_FILE_LOGGING', 'false')}")
logger.info(f"Python バージョン: {sys.version}")
logger.info(f"OpenCV バージョン: {cv2.__version__}")
logger.info(f"NumPy バージョン: {np.__version__}")

# 一時フォルダのパス設定
TMP_DIR = "tmp"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

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
    openai.api_config_type = "azure"  # 明示的にapi_config_typeを設定
    openai.api_key = azure_api_key
    openai.api_base = azure_endpoint
    openai.api_version = azure_api_version
    default_model = azure_deployment
else:
    # 環境変数からOpenAI APIキーを取得
    logger.info("OpenAI APIを使用します")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_type = "openai"  # 明示的にapi_typeを設定
    openai.api_config_type = "openai"  # 明示的にapi_config_typeを設定
    default_model = "gpt-4o"  # デフォルトモデル
    
    if not openai.api_key:
        st.error("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")
        st.stop()

# NumPy型をJSON変換用にカスタマイズする関数を定義
def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

##############################
# LLM呼び出しのユーティリティ #
##############################
def call_gpt(prompt, image_base64_list=None, max_tokens=1500, temperature=0.0, model=None):
    """
    LLMを呼び出すユーティリティ関数
    Args:
        prompt: プロンプトテキスト
        image_base64_list: Base64エンコードされた画像のリスト（複数画像対応）
        max_tokens: 最大トークン数
        temperature: 温度パラメータ
        model: 使用するモデル（Noneの場合はデフォルトモデルを使用）
    """
    # モデルが指定されていない場合はデフォルトモデルを使用
    if model is None:
        model = default_model
        
    messages = [
        {"role": "system", "content": "あなたは与えられた画像から俯瞰的な目線で正確に文書構造を読み取り、JSON形式で返してください。"}
    ]

    logger.info("==========call_gpt start==========")
    logger.info(f"使用モデル: {model}")
    
    if image_base64_list:
        # 画像が1つ以上ある場合
        logger.info(f"image_base64_list:{len(image_base64_list)}")
        
        # Azure OpenAI APIとOpenAI APIで画像の扱いが異なる
        if openai.api_type == "azure":
            # Azure OpenAI APIの場合
            content = []
            content.append({"type": "text", "text": prompt})
            for img_base64 in image_base64_list:
                logger.info(f"img_base64:{len(img_base64)}")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # 通常のOpenAI APIの場合
            content = [{"type": "text", "text": prompt}]
            for img_base64 in image_base64_list:
                logger.info(f"img_base64:{len(img_base64)}")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            messages.append({
                "role": "user",
                "content": content
            })
    else:
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

##############################
# PDFをページ単位で画像変換 #
##############################
def pdf_to_images(pdf_file, dpi=1000, max_pixels=4000000):
    try:
        logger.info(f"PDFの画像変換とテキスト抽出を開始: DPI={dpi}")
        
        # セッション固有の一時フォルダを作成
        session_id = str(uuid.uuid4())
        session_tmp_dir = os.path.join(TMP_DIR, session_id)
        os.makedirs(session_tmp_dir, exist_ok=True)
        logger.info(f"一時フォルダを作成: {session_tmp_dir}")
        
        # PDFファイルの内容をバイトとして読み込む
        pdf_bytes = pdf_file.read()
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        total_pages = len(doc)
        logger.info(f"PDFの総ページ数: {total_pages}ページ")
        
        images_info = []
        for page_num in range(total_pages):
            logger.info(f"ページ {page_num + 1}/{total_pages} を処理中...")
            page = doc.load_page(page_num)
            
            # テキスト抽出
            text = page.get_text()
            
            # 画像変換
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # 早い段階でのリサイズを適用（メモリ使用量削減）
            img_np = np.array(img)
            img_np = resize_with_aspect_ratio(img_np, max_pixels=max_pixels)
            img = Image.fromarray(img_np)
            
            # 画像を一時フォルダに保存
            img_path = os.path.join(session_tmp_dir, f"page_{page_num + 1}.jpg")
            img.save(img_path, "JPEG", quality=95)
            
            images_info.append({
                "image": img,
                "path": img_path,
                "text": text,
                "tmp_dir": session_tmp_dir  # 一時フォルダのパスを追加
            })
        
        doc.close()
        logger.info("PDFの画像変換とテキスト抽出が完了しました")
        return images_info
        
    except Exception as e:
        logger.error(f"PDFの処理に失敗: {str(e)}")
        st.error(f"PDFの処理に失敗しました: {str(e)}")
        return None

def get_page_image(doc, page_num, dpi=1000, max_pixels=4000000, session_tmp_dir=None):
    """
    1ページだけを処理して画像を返す関数（オンデマンド処理）
    Args:
        doc: PDFドキュメントオブジェクト
        page_num: 処理するページ番号（0始まり）
        dpi: 画像変換のDPI値
        max_pixels: 最大ピクセル数
        session_tmp_dir: 一時フォルダのパス（Noneの場合は新規作成）
    Returns:
        dict: 画像情報を含む辞書
    """
    try:
        logger.info(f"ページ {page_num + 1} のオンデマンド処理を開始")
        
        # 一時フォルダが指定されていない場合は新規作成
        if session_tmp_dir is None or not os.path.exists(session_tmp_dir):
            session_id = str(uuid.uuid4())
            session_tmp_dir = os.path.join(TMP_DIR, session_id)
            logger.info(f"オンデマンド処理用の一時フォルダを作成: {session_tmp_dir}")
        
        # 一時フォルダが存在することを確認
        os.makedirs(session_tmp_dir, exist_ok=True)
        logger.info(f"一時フォルダの確認または作成: {session_tmp_dir}")
        
        # ページを読み込み
        page = doc.load_page(page_num)
        
        # テキスト抽出
        text = page.get_text()
        
        # 画像パスを設定
        img_path = os.path.join(session_tmp_dir, f"page_{page_num + 1}.jpg")
        
        # 既にファイルが存在する場合はそれを使用
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            logger.info(f"既存の画像ファイルを使用: {img_path}")
            img = Image.open(img_path)
            return {
                "image": img,
                "path": img_path,
                "text": text,
                "tmp_dir": session_tmp_dir
            }
        
        # 画像変換
        logger.info(f"ページ {page_num + 1} の画像変換を開始（DPI={dpi}）")
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 早い段階でのリサイズを適用
        logger.info(f"ページ {page_num + 1} の画像リサイズを実行（最大ピクセル数={max_pixels}）")
        img_np = np.array(img)
        img_np = resize_with_aspect_ratio(img_np, max_pixels=max_pixels)
        img = Image.fromarray(img_np)
        
        # 画像を一時フォルダに保存
        logger.info(f"ページ {page_num + 1} の画像を保存: {img_path}")
        img.save(img_path, "JPEG", quality=95)
        
        # 保存された画像の存在を確認
        if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
            raise IOError(f"画像ファイルの保存に失敗しました: {img_path}")
        
        logger.info(f"ページ {page_num + 1} のオンデマンド処理が完了")
        return {
            "image": img,
            "path": img_path,
            "text": text,
            "tmp_dir": session_tmp_dir
        }
        
    except Exception as e:
        logger.error(f"ページ {page_num + 1} のオンデマンド処理に失敗: {str(e)}", exc_info=True)
        # エラーが発生した場合でも最低限の情報を返すようにする
        if 'page' in locals() and 'text' not in locals():
            try:
                text = page.get_text()
            except:
                text = ""
                
        return {
            "image": None,
            "path": None,
            "text": text if 'text' in locals() else "",
            "tmp_dir": session_tmp_dir,
            "error": str(e)
        }

def encode_image_to_base64(image_info):
    """
    画像をBase64エンコードされた文字列に変換
    Args:
        image_info (dict): 画像情報（画像オブジェクトとファイルパス）
    Returns:
        str: Base64エンコードされた画像データ
    """
    try:
        # 保存された画像ファイルを読み込んでBase64エンコード
        with open(image_info["path"], "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        st.error(f"画像のエンコードに失敗しました: {str(e)}")
        return None

def cleanup_tmp_files():
    """
    一時フォルダとその中のファイルを削除
    """
    try:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
            os.makedirs(TMP_DIR)
            logger.info("一時フォルダを清掃しました")
    except Exception as e:
        logger.error(f"一時フォルダの清掃に失敗: {str(e)}")

def resize_with_aspect_ratio(image, max_pixels=4000000):
    """
    アスペクト比を保持しながら、指定された最大ピクセル数以下になるようにリサイズする関数
    Args:
        image: 入力画像
        max_pixels: 最大ピクセル数（デフォルト4MP）
    Returns:
        リサイズされた画像
    """
    height, width = image.shape[:2]
    current_pixels = height * width
    
    if current_pixels <= max_pixels:
        return image
    
    scale = np.sqrt(max_pixels / current_pixels)
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

#############################################################
# マルチパスで「トップレベルの章節」抽出→ページ範囲の要約付与 #
#############################################################
def create_section_summary(section, pages_images, total_pages, file_name, current_depth=1, max_summary_depth=2, max_content_depth=2):
    """
    再帰的にセクションの要約を作成する関数
    Args:
        section: セクション情報
        pages_images: 全ページの画像情報
        total_pages: 総ページ数
        current_depth: 現在の階層の深さ
        max_summary_depth: 要約を作成する最大の階層の深さ
        max_content_depth: コンテンツを抽出する最大の階層の深さ
    """
    s_page = section.get("start_image", 0) - 1
    e_page = s_page + total_pages - 1
    
    s_page = max(0, s_page)
    e_page = max(0, e_page)
    if e_page < s_page:
        e_page = s_page

    section_images = pages_images[s_page:e_page+1]
    section_images_base64 = [encode_image_to_base64(image) for image in section_images]
    
    section_text = "\n".join([image["text"] for image in section_images])
    section_text_with_row = "\n".join([f"row_{i+1}:{line}" for i, line in enumerate(section_text.split("\n"))])

    # 要約の作成（深さの条件チェック）
    if current_depth <= max_summary_depth:
        sum_prompt = f"""これはあるドキュメントの「{section.get('title', 'NoTitle')}」セクションを含む画像です。
        このセクションが何について記載されているか概要を説明してください。回答はJSON形式で"summary"キーに要約を含めてください。
        またこのセクションに図や表や画像が含まれている場合は、"images"キーにそれらの概要を含めてください。図や表や画像がない場合は"images"キーは空白で回答してください。
        「{section.get('title', 'NoTitle')}」セクションが見つからない場合は空白で回答してください。

        """
        summary = call_gpt(sum_prompt, image_base64_list=section_images_base64)
        try:
            summary_obj = json.loads(summary)
            section["summary"] = summary_obj.get("summary", "")
            if summary_obj.get("images", "") != "":
                section["images"] = summary_obj.get("images", "") 
        except:
            section["summary"] = ""
    else:
        # summary_depthを超える階層ではsummaryを設定しない
        section.pop("summary", None)

    # コンテンツの抽出（深さの条件チェック）
    if current_depth <= max_content_depth:
        content_prompt = f"""これはあるドキュメントの「{section.get('title', 'NoTitle')}」セクションを含むテキストです。
        このセクションがどの行からどの行までか回答してください。回答はJSON形式で"from_row"、"to_row"キーを含めてください。
        「{section.get('title', 'NoTitle')}」セクションが見つからない場合は空白で回答してください。
        回答フォーマット：
        {{
          "from_row": "row_1",
          "to_row": "row_10"
        }}

        テキスト：
        {section_text_with_row}
        """
        content = call_gpt(content_prompt)
        content_obj = json.loads(content)
        
        from_row = content_obj.get("from_row", "")
        to_row = content_obj.get("to_row", "")
        
        from_row_num = int(from_row.replace("row_", "")) if from_row else 1
        to_row_num = int(to_row.replace("row_", "")) if to_row else len(section_text_with_row.split("\n"))
        
        section_lines = section_text.split("\n")
        section_content = "\n".join(section_lines[from_row_num-1:to_row_num])
        section["content"] = section_content

    else:
        # content_depthを超える階層ではcontentを設定しない
        section.pop("content", None)

    if current_depth == max_content_depth:
        # s_pageからe_page
        if section.get("images", "") != "":
            
            # 図や表の抽出
            prompt = f"""これはあるドキュメントの「{section.get('title', 'NoTitle')}」セクションを含む画像です。
        このセクションに含まれているグラフ、図や表の情報をJSON形式で正確に抽出してください。以下に添付するテキスト情報も参考にしてください。
        回答はJSON形式で"objects"キーにそれらのグラフ、図や表の情報を含めてください。
        「{section.get('title', 'NoTitle')}」セクションが見つからない場合は空白で回答してください。

        テキスト：
        {section_content}
        """
            # 画像情報の取得
            image_base64_list_sub = []
            try:
                # analysis_results.jsonから画像情報を取得
                file_dir = os.path.join(TMP_DIR, "analysis_results", file_name)
                json_path = os.path.join(file_dir, "analysis_results.json")
                
                st.write("============section============")
                st.json(section)
                st.write("============prompt text============")
                st.write(prompt)
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        analysis_results = json.load(f)
                    
                    st.write(f"============analysis_results:{section.get('title', 'NoTitle')} {s_page}ページ → {e_page}ページ============")
                    
                    # start_pageからend_pageまでの画像を取得  
                    for page_num in range(s_page + 1, e_page + 2):
                        if "pages" in analysis_results and str(page_num) in analysis_results["pages"]:
                            for page_info in analysis_results["pages"][str(page_num)]["regions"]:
                                path = page_info["path"]
                                type = page_info["type"]
                                # typeに「図」「イメージ」「表」「グラフ」が含まれているかチェック
                                if "図" in type or "イメージ" in type or "表" in type or "グラフ" in type:
                                    st.write(f"============prompt image:{page_num}============")
                                    if os.path.exists(path):
                                        with open(path, "rb") as img_file:
                                            st.image(path)
                                            img_base64 = base64.b64encode(img_file.read()).decode()
                                        image_base64_list_sub.append(img_base64)
            
                logger.info(f"セクション画像を取得: {s_page}ページ → {e_page}ページ ({len(image_base64_list_sub)}枚)")
            except Exception as e:
                logger.error(f"セクション画像の取得中にエラー: {str(e)}")
                image_base64_list_sub = []
            objects = call_gpt(prompt, image_base64_list=image_base64_list_sub)
            objects_obj = json.loads(objects)
            section["objects"] = objects_obj.get("objects", "")
    
    # サブセクションの再帰的処理
    if "sections" in section and isinstance(section["sections"], list):
        for subsec in section["sections"]:
            # サブセクションの開始ページを決定
            s_page = subsec.get("start_image", 0) - 1
            # サブセクションの終了ページを決定
            # デフォルトは親セクションの終了ページ
            e_page = section.get("end_image", total_pages)
            
            # サブセクションの終了ページを、このサブセクションの次のサブセクションの開始ページから決定
            # 現在のサブセクションのインデックスを取得
            current_index = section["sections"].index(subsec)
            
            # 次のサブセクションが存在する場合
            if current_index + 1 < len(section["sections"]):
                next_subsec = section["sections"][current_index + 1]
                if "start_image" in next_subsec:
                    # 次のサブセクションの開始ページの1つ前をこのサブセクションの終了ページとする
                    e_page = next_subsec["start_image"] - 1
                        
            st.write(f"============subsec:{subsec.get('title', 'NoTitle')} {s_page}ページ → {e_page}ページ============")
            create_section_summary(subsec, pages_images, e_page-s_page+1, file_name,
                                current_depth + 1, max_summary_depth, max_content_depth)

def extract_and_summarize_level2_sections_multipass(pages_images, chunk_size=10, summary_depth=2, content_depth=2, file_name=None):
    logger.info(f"レベル2セクションの抽出を開始: 総ページ数={len(pages_images)}, チャンクサイズ={chunk_size}")
    partial_section_lists = []
    raw_sections = ""
    total_pages = len(pages_images)

    for start in range(0, total_pages, chunk_size):
        chunk_start = max(0, start - 1) 
        chunk_end = min(start + chunk_size, total_pages)
        
        logger.info(f"チャンク処理中: {chunk_start+1}ページ → {chunk_end}ページ")
        chunk_images = pages_images[chunk_start:chunk_end]
        
        # チャンク内の画像とテキストを処理
        chunk_images_base64 = []
        chunk_texts = []
        for image in chunk_images:
            img_base64 = encode_image_to_base64(image)
            if img_base64:
                chunk_images_base64.append(img_base64)
                chunk_texts.append(image["text"])

        if not chunk_images_base64:
            logger.warning(f"チャンク {chunk_start+1}-{chunk_end} の画像エンコードに失敗")
            continue

        logger.info(f"GPTにチャンク {chunk_start+1}-{chunk_end} の解析中...")
        prompt = f"""
以下の画像のドキュメントの目次を作成するために、以下の情報を抽出してください。
- 階層のレベルを"level"キーに含めてください。level1が最も上位の章・節で、level2がその下位の節、level3以降はさらにその下位の条や項番など。
- 出力のJSONは"sections"をキーに以下の形式としてください:
  [
    {{
      "level": <1/2/3...>
      "title": "章や節などのタイトル。項番号がある場合は省略せずに含めてください。",
      "start_image": "{len(chunk_images_base64)}枚の画像のうち何枚目の画像にこのtitleが含まれているか数字で回答。"
    }},
    ...
  ]

"""
        
        raw_sections = call_gpt(prompt, image_base64_list=chunk_images_base64)
        try:
            sections_obj = json.loads(raw_sections)["sections"]
            if isinstance(sections_obj, dict):
                sections_obj = [sections_obj]
            # start_pageにchunk_startを足す
            for section in sections_obj:
                if "start_image" in section:
                    section["start_image"] = section["start_image"] + chunk_start
            partial_section_lists.append(sections_obj)
        except:
            continue

    # (2) すべての「部分セクション情報」を連結し、LLMに再度依頼してマージ
    merged_input = []
    for partial in partial_section_lists:
        merged_input.extend(partial)

    # マージ前のセクション数を記録
    merge_prompt = f"""以下は、複数チャンクから抽出したドキュメントの構造情報の暫定リストです。
一つのPDF全体として考えたときに、重複やページ範囲の重なり・粒度ズレを調整して、最終的なセクション構造を作ってください。
- 同じレベルの節が複数回出てきますので、重複削除してください。
- ただマージするのではなく、項番号などから階層構造をイメージしてlevelを調整してください。
- titleは項番号も省略せずに含めてください。項番号は絶対に変更しないでください。
- "start_image"は変更しないでください。
- "sections"というキーに、"title","start_image"のキーを持つ階層構造をJSONで返してください

暫定リスト:
{json.dumps(merged_input, ensure_ascii=False, indent=2)}
"""
    merged_raw = call_gpt(merge_prompt)
    # merged_raw = call_gpt(merge_prompt, image_base64_list=[encode_image_to_base64(image) for image in pages_images])

    try:
        final_sections = json.loads(merged_raw)["sections"]
        if isinstance(final_sections, dict):
            final_sections = [final_sections]
    except:
        final_sections = merged_input

    # 実際のセクション階層の深さを計算
    def get_max_depth(sections, current_depth=1):
        max_depth = current_depth
        for section in sections:
            if "sections" in section:
                sub_depth = get_max_depth(section["sections"], current_depth + 1)
                max_depth = max(max_depth, sub_depth)
        return max_depth

    # 実際の階層の深さを取得
    actual_max_depth = get_max_depth(final_sections)
    
    # 指定された深さが実際の深さを超えないように調整
    summary_depth = min(summary_depth, actual_max_depth)
    content_depth = min(content_depth, actual_max_depth)
    
    logger.info(f"実際のセクション階層の深さ: {actual_max_depth}")

    # 各セクションの要約とコンテンツを作成
    for section in final_sections:
        # セクションの開始ページと終了ページを決定
        s_page = section.get("start_image", 0) - 1
        st.write(f"============section:{section.get('title', 'NoTitle')} {s_page}ページ============")
        # 次のセクションの開始ページ-1 または 最終ページをend_pageとする
        next_section_start = None
        for next_section in final_sections:
            if next_section.get("start_image", 0) > section.get("start_image", 0):
                next_section_start = next_section.get("start_image", 0)
                st.write(f"============next_section:{next_section.get('title', 'NoTitle')} {next_section_start}ページ============")
                break
        st.write(f"============total_pages:{total_pages}============")  
        e_page = (next_section_start - 2) if next_section_start else (total_pages - 1)
        st.write(f"============e_page:{e_page}============")
        # ページ範囲を正規化
        s_page = max(0, min(s_page, total_pages-1))
        e_page = max(0, min(e_page, total_pages-1))
        if e_page < s_page:
            e_page = s_page

        # セクションに関連する画像のみを抽出
        section_images = pages_images[s_page:e_page+1]

        st.write(f"============section_images:{section.get('title', 'NoTitle')} {s_page}ページ → {e_page}ページ============")
        
        # セクションの要約とコンテンツを作成（対象画像のみを渡す）
        create_section_summary(section, section_images, e_page-s_page+1, file_name, 1, summary_depth, content_depth)

    logger.info("レベル2セクションの抽出が完了")
    return final_sections

##########################
# Streamlitのメイン処理  #
##########################
def main():
    st.title("PDF分析アプリ")
    st.sidebar.title("設定")

    # 一時フォルダクリアボタンをサイドバーに追加
    if st.sidebar.button("一時フォルダをクリア"):
        cleanup_tmp_files()
        st.sidebar.success("一時フォルダをクリアしました")

    # DPI設定をサイドバーに追加
    dpi_value = st.sidebar.slider("DPI設定", min_value=100, max_value=1000, value=300, step=100)
    max_pixels = st.sidebar.slider("最大ピクセル数（百万単位）", min_value=1, max_value=10, value=4, step=1) * 1000000
    
    # セッション状態に値を保存（image_analysis.pyで使用するため）
    st.session_state.dpi_value = dpi_value
    st.session_state.max_pixels = max_pixels

    # ファイルアップロード
    uploaded_file = st.file_uploader("PDFファイルをアップロード", type=['pdf'])

    if uploaded_file is not None:
        try:
            # ファイル名を取得（拡張子を除く）
            file_name = os.path.splitext(uploaded_file.name)[0]
            
            # PDFドキュメントをセッションに保存（オンデマンド処理のため）
            if 'pdf_doc' not in st.session_state or st.session_state.get('current_pdf') != uploaded_file.name:
                pdf_bytes = uploaded_file.read()
                pdf_stream = io.BytesIO(pdf_bytes)
                st.session_state.pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
                st.session_state.current_pdf = uploaded_file.name
                st.session_state.total_pages = len(st.session_state.pdf_doc)
                
                # セッション固有の一時フォルダを作成
                st.session_state.session_tmp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
                os.makedirs(st.session_state.session_tmp_dir, exist_ok=True)
                logger.info(f"セッション一時フォルダを作成: {st.session_state.session_tmp_dir}")
                
                # ページ情報の初期化
                st.session_state.pages_images = []
            
            # タブを作成
            # tab1, tab2= st.tabs(["文書構造分析", "画像分析"])
            # 画像分析タブを非表示にして文書構造分析のみ表示
            # with tab1:
            st.header("文書構造分析")
            
            # 処理方法の選択
            processing_method = st.radio(
                "処理方法を選択",
                ["全ページ一括処理", "オンデマンド処理（メモリ効率）"],
                index=1  # デフォルトはオンデマンド処理
            )
            
            # 設定
            chunk_size = st.sidebar.slider("チャンクサイズ", min_value=1, max_value=20, value=10)
            summary_depth = st.sidebar.slider("要約を作成する階層の深さ", min_value=1, max_value=5, value=2)
            content_depth = st.sidebar.slider("コンテンツを抽出する階層の深さ", min_value=1, max_value=5, value=2)

            if st.button("文書構造を分析"):
                with st.spinner("文書構造を分析中..."):
                    if processing_method == "全ページ一括処理":
                        # 従来の処理方法（全ページ一括）
                        uploaded_file.seek(0)  # ファイルポインタをリセット
                        pages_images = pdf_to_images(uploaded_file, dpi=dpi_value, max_pixels=max_pixels)
                        if not pages_images:
                            st.error("PDFの処理に失敗しました。")
                            return
                    else:
                        # オンデマンド処理（必要なページのみ）
                        pages_images = []
                        progress_bar = st.progress(0)
                        for page_num in range(st.session_state.total_pages):
                            progress_percent = (page_num + 1) / st.session_state.total_pages
                            progress_bar.progress(progress_percent)
                            # st.write(f"ページ {page_num + 1}/{st.session_state.total_pages} を処理中...")
                            
                            page_image = get_page_image(
                                st.session_state.pdf_doc,
                                page_num,
                                dpi=dpi_value,
                                max_pixels=max_pixels,
                                session_tmp_dir=st.session_state.session_tmp_dir
                            )
                            if page_image:
                                pages_images.append(page_image)
                                
                                # メモリを解放するため、画像オブジェクトは保持せずパスのみ保存
                                if "image" in page_image:
                                    page_image["image"].close()
                                    page_image["image"] = None
                        
                        # 文書構造分析を実行
                        result = extract_and_summarize_level2_sections_multipass(
                            pages_images,
                            chunk_size=chunk_size,
                            summary_depth=summary_depth,
                            content_depth=content_depth,
                            file_name=file_name
                        )
                        
                        # JSONファイルを出力
                        json_output_path = os.path.join("output", file_name, "document_structure.json")
                        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
                        with open(json_output_path, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        st.success(f"文書構造分析結果をJSONファイルに保存しました: {json_output_path}")
                        st.json(result)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"アプリケーションエラー: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()