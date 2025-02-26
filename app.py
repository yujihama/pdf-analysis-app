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

# .envファイルの読み込み
load_dotenv()

# ロギングの設定
handlers = [logging.StreamHandler(sys.stdout)]

# 必要な場合のみログファイルを追加
if os.getenv('ENABLE_FILE_LOGGING', 'false').lower() == 'true':
    handlers.append(
        logging.FileHandler(f'pdf_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

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
        このセクションに含まれている図や表の情報をJSON形式で正確に抽出してください。以下に添付するテキスト情報も参考にしてください。
        回答はJSON形式で"objects"キーにそれらの図や表の情報を含めてください。
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

    # ページ分析マネージャーをセッションステートとして初期化
    if 'page_manager' in st.session_state:
        del st.session_state.page_manager  # 古いインスタンスを削除
    st.session_state.page_manager = PageAnalysisManager()

    # 一時フォルダクリアボタンをサイドバーに追加
    if st.sidebar.button("一時フォルダをクリア"):
        cleanup_tmp_files()
        st.sidebar.success("一時フォルダをクリアしました")

    # DPI設定をサイドバーに追加
    dpi_value = st.sidebar.slider("DPI設定", min_value=100, max_value=1000, value=300, step=100)
    max_pixels = st.sidebar.slider("最大ピクセル数（百万単位）", min_value=1, max_value=10, value=4, step=1) * 1000000

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
            tab1, tab2= st.tabs(["文書構造分析", "画像分析"])

            with tab1:
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
                                st.write(f"ページ {page_num + 1}/{st.session_state.total_pages} を処理中...")
                                
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

            with tab2:
                st.header("画像分析")
                st.write("PDFの各ページから表や図、グラフを抽出します。")
                
                # 分析ボタン
                if st.button("画像分析を実行"):
                    if not hasattr(st.session_state, 'pdf_doc') or st.session_state.pdf_doc is None:
                        st.error("PDFファイルが読み込まれていません。先にPDFファイルをアップロードしてください。")
                        return
                    
                    # 一時フォルダの状態を確認
                    if not hasattr(st.session_state, 'session_tmp_dir') or not os.path.exists(st.session_state.session_tmp_dir):
                        st.session_state.session_tmp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
                        os.makedirs(st.session_state.session_tmp_dir, exist_ok=True)
                        logger.info(f"画像分析用の一時フォルダを新規作成: {st.session_state.session_tmp_dir}")
                    else:
                        logger.info(f"既存の一時フォルダを使用: {st.session_state.session_tmp_dir}")
                    
                    total_pages = st.session_state.total_pages
                    progress_bar = st.progress(0)
                    
                    # PageAnalysisManagerを初期化
                    if 'page_manager' not in st.session_state:
                        st.session_state.page_manager = PageAnalysisManager()
                    
                    for page_num in range(total_pages):
                        with st.spinner(f"ページ {page_num + 1}/{total_pages} を分析中..."):
                            # オンデマンドでページ画像を取得
                            page_image = get_page_image(
                                st.session_state.pdf_doc,
                                page_num,
                                dpi=dpi_value,
                                max_pixels=max_pixels,
                                session_tmp_dir=st.session_state.session_tmp_dir
                            )
                            
                            if page_image:
                                # 連結成分ベースの分析を実行
                                analysis_result = analyze_image_content_with_connected_components(
                                    page_image["path"],
                                    file_name,
                                    page_num + 1
                                )
                                
                                # 結果を表示
                                with st.expander(f"ページ {page_num + 1} の分析結果"):
                                    display_single_analysis_results(
                                        analysis_result,
                                        st.session_state.page_manager,
                                        page_num + 1,
                                        file_name
                                    )
                                
                                # 使用後は画像オブジェクトを解放（メモリ効率化）
                                if "image" in page_image and page_image["image"] is not None:
                                    page_image["image"].close()
                                    page_image["image"] = None
                            else:
                                st.warning(f"ページ {page_num + 1} の画像取得に失敗しました。スキップします。")
                            
                            # プログレスバーを更新
                            progress_bar.progress((page_num + 1) / total_pages)
                    
                    st.success("すべてのページの分析が完了しました")
                    
                    # 最終的なJSONファイルの内容を表示
                    json_output_dir = os.path.join("output", file_name)
                    json_filename = os.path.join(json_output_dir, "analysis_results.json")
                    os.makedirs(json_output_dir, exist_ok=True)
                    
                    # 分析結果をJSONに保存
                    all_results = st.session_state.page_manager.get_all_results()
                    with open(json_filename, "w", encoding="utf-8") as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    
                    st.write("### 全ページの分析結果")
                    st.json(all_results)
                
                # 個別ページの分析オプションも追加
                st.divider()
                st.subheader("個別ページの分析")
                
                if hasattr(st.session_state, 'pdf_doc') and st.session_state.pdf_doc:
                    # ページ選択
                    page_num = st.selectbox(
                        "分析するページを選択",
                        options=list(range(1, st.session_state.total_pages + 1)),
                        format_func=lambda x: f"ページ {x}"
                    )
                    
                    # 個別分析ボタン
                    if st.button("解析実行"):
                        with st.spinner("画像を分析中..."):
                            # 一時フォルダの状態を確認
                            if not hasattr(st.session_state, 'session_tmp_dir') or not os.path.exists(st.session_state.session_tmp_dir):
                                st.session_state.session_tmp_dir = os.path.join(TMP_DIR, str(uuid.uuid4()))
                                os.makedirs(st.session_state.session_tmp_dir, exist_ok=True)
                                logger.info(f"個別ページ分析用の一時フォルダを新規作成: {st.session_state.session_tmp_dir}")
                            
                            # オンデマンドでページ画像を取得
                            selected_image = get_page_image(
                                st.session_state.pdf_doc,
                                page_num - 1,
                                dpi=dpi_value,
                                max_pixels=max_pixels,
                                session_tmp_dir=st.session_state.session_tmp_dir
                            )
                            
                            if selected_image:
                                # 連結成分ベースの分析を実行
                                analysis_result = analyze_image_content_with_connected_components(
                                    selected_image["path"],
                                    file_name,
                                    page_num
                                )
                                
                                # 結果を表示
                                st.subheader("検出結果")
                                display_single_analysis_results(
                                    analysis_result,
                                    st.session_state.page_manager,
                                    page_num,
                                    file_name
                                )
                                
                                # 使用後は画像オブジェクトを解放
                                if "image" in selected_image and selected_image["image"] is not None:
                                    selected_image["image"].close()
                                    selected_image["image"] = None
                            else:
                                st.error(f"ページ {page_num} の画像取得に失敗しました。再試行するか、DPI値を下げてみてください。")
                                if "error" in selected_image:
                                    st.error(f"エラー詳細: {selected_image['error']}")
                else:
                    st.info("PDFファイルをアップロードすると、個別ページの分析が可能になります。")
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"アプリケーションエラー: {str(e)}", exc_info=True)

class PageAnalysisManager:
    """
    各ページの分析結果を管理するクラス
    """
    def __init__(self):
        self.page_results = {}  # {page_num: analysis_result}
        self.merged_regions = {}  # {page_num: merged_regions_info}
        
    def add_page_result(self, page_num, analysis_result, llm_analysis=None):
        """
        ページの分析結果を追加
        Args:
            page_num: ページ番号
            analysis_result: 分析結果
            llm_analysis: LLMによる分析結果
        """
        if page_num not in self.page_results:
            self.page_results[page_num] = {
                "original_result": analysis_result,
                "llm_analysis": llm_analysis,
                "regions": []
            }
            
        # 領域情報を抽出
        regions_info = []
        for figure in analysis_result["figures"]:
            # NumPy型の値を標準のPython型に変換
            region_info = {
                "id": figure["id"],
                "bbox": {
                    "x": int(figure["position"]["x"]),
                    "y": int(figure["position"]["y"]),
                    "width": int(figure["position"]["width"]),
                    "height": int(figure["position"]["height"])
                },
                "merged_from": figure.get("merged_from", []),
                "path": figure["path"]
            }
            
            # LLM分析結果がある場合は追加情報を付与
            if llm_analysis and "error" not in llm_analysis:
                region_data = next(
                    (r for r in llm_analysis["regions"] if r["id"] == figure["id"]),
                    None
                )
                if region_data:
                    region_info.update({
                        "type": region_data["type"],
                        "description": region_data["description"]
                    })
            
            regions_info.append(region_info)
        
        # 統合提案がある場合は統合後の情報を生成
        if llm_analysis and "error" not in llm_analysis and llm_analysis.get("merge_suggestions"):
            merged_regions = []
            used_regions = set()
            
            # 統合提案に基づいて領域を統合
            for suggestion in llm_analysis["merge_suggestions"]:
                group = suggestion["group"]
                if not group:
                    continue
                
                # グループ内の領域を取得
                regions_to_merge = [
                    r for r in regions_info
                    if r["id"] in group
                ]
                
                if not regions_to_merge:
                    continue
                
                # 統合領域の範囲を計算（NumPy型の値を標準のPython型に変換）
                min_x = int(min(r["bbox"]["x"] for r in regions_to_merge))
                min_y = int(min(r["bbox"]["y"] for r in regions_to_merge))
                max_x = int(max(r["bbox"]["x"] + r["bbox"]["width"] for r in regions_to_merge))
                max_y = int(max(r["bbox"]["y"] + r["bbox"]["height"] for r in regions_to_merge))
                
                # 新しい統合領域のID（グループの最小IDを使用）
                new_id = min(r["id"] for r in regions_to_merge)
                
                # 統合領域の情報を作成
                merged_region = {
                    "id": int(new_id),
                    "bbox": {
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x,
                        "height": max_y - min_y
                    },
                    "merged_from": group,
                    "path": os.path.join(os.path.dirname(next(r["path"] for r in regions_to_merge if r["id"] == new_id)), f"merged_region_{new_id}.png"),
                    "type": next((r.get("type", "不明") for r in regions_to_merge if r["id"] == new_id), "不明"),
                    "description": next((r.get("description", "") for r in regions_to_merge if r["id"] == new_id), ""),
                    "merge_reason": suggestion.get("reason", "")
                }
                
                merged_regions.append(merged_region)
                used_regions.update(group)
            
            # 統合されなかった領域を追加
            for region in regions_info:
                if region["id"] not in used_regions:
                    merged_regions.append(region)
            
            # 統合後の情報を保存
            self.merged_regions[page_num] = {
                "regions": merged_regions,
                "total_regions": len(merged_regions),
                "merged_groups": len([r for r in merged_regions if r.get("merged_from")])
            }
        
        # 元の領域情報を保存
        self.page_results[page_num]["regions"] = regions_info
    
    def get_page_result(self, page_num):
        """
        指定ページの分析結果を取得
        """
        return self.page_results.get(page_num)
    
    def get_merged_regions_info(self, page_num):
        """
        指定ページの統合後の領域情報を取得
        Args:
            page_num: ページ番号
        Returns:
            dict: 統合後の領域情報。存在しない場合はNone
        """
        return self.merged_regions.get(page_num)
    
    def get_all_results(self):
        """
        全ページの分析結果を取得（統合前後の両方）
        """
        results = {}
        for page_num in self.page_results.keys():
            results[page_num] = {
                "original": self.page_results[page_num],
                "merged": self.merged_regions.get(page_num)
            }
        return results
    
    def get_regions_summary(self):
        """
        全ページの領域サマリーを取得（統合後の情報を優先）
        """
        summary = {}
        for page_num in self.page_results.keys():
            if page_num in self.merged_regions:
                merged_info = self.merged_regions[page_num]
                summary[page_num] = {
                    "total_regions": merged_info["total_regions"],
                    "merged_regions": merged_info["merged_groups"],
                    "regions": merged_info["regions"]
                }
            else:
                original_info = self.page_results[page_num]
                summary[page_num] = {
                    "total_regions": len(original_info["regions"]),
                    "merged_regions": len([r for r in original_info["regions"] if r["merged_from"]]),
                    "regions": original_info["regions"]
                }
        return summary
    
def display_single_analysis_results(analysis_result, page_manager, page_num, file_name):
    """個別の分析結果を表示する関数"""
    if "error" in analysis_result:
        st.error(analysis_result["error"])
        return

    # 検出結果の表示
    st.image(analysis_result["result_image"], caption="検出された要素", use_container_width=True)

    # 検出された要素の数を表示
    num_figures = len(analysis_result["figures"])
    st.write(f"検出された領域: {num_figures}件")

    # 基本の分析結果を作成（NumPy型を標準のPython型に変換）
    base_result = {
        "total_regions": int(num_figures),
        "merged_regions": 0,
        "regions": [
            {
                "id": str(fig["id"]),
                "bbox": {
                    "x": int(fig["position"]["x"]),
                    "y": int(fig["position"]["y"]),
                    "width": int(fig["position"]["width"]),
                    "height": int(fig["position"]["height"])
                },
                "type": fig.get("type", "未分析"),
                "description": fig.get("description", ""),
                "path": str(fig["path"]),
                "parent_id": fig.get("parent_id"),
                "split_depth": fig.get("split_depth", 0),
                "merged_from": fig.get("merged_from", [])
            }
            for fig in analysis_result["figures"]
        ]
    }

    # LLMによる分析を実行
    if analysis_result["figures"]:
        # result_imageを概要画像として渡す
        llm_analysis = analyze_regions_with_llm(
            analysis_result["figures"], 
            file_name, 
            page_num,
            overview_path=analysis_result["result_image"]  # 検出結果の可視化画像を概要画像として使用
        )
        
        if isinstance(llm_analysis, str):
            try:
                llm_analysis = json.loads(llm_analysis)
            except json.JSONDecodeError as e:
                logger.error(f"LLM分析結果のJSONパースに失敗: {str(e)}")
                llm_analysis = {"error": "分析結果の解析に失敗しました"}
        
        # st.write("LLMの分析結果")
        # st.json(llm_analysis)

        if "error" not in llm_analysis:
            # LLMの分析結果をbase_resultに反映
            if "regions" in llm_analysis:
                # IDをキーとしたLLM分析結果の辞書を作成
                llm_regions_dict = {
                    region["id"]: region 
                    for region in llm_analysis["regions"]
                }
                
                # base_resultの各領域を更新
                for region in base_result["regions"]:
                    if region["id"] in llm_regions_dict:
                        llm_region = llm_regions_dict[region["id"]]
                        region.update({
                            "type": llm_region.get("type", "未分析"),
                            "description": llm_region.get("description", ""),
                            "needs_split": llm_region.get("needs_split", False),
                            "split_reason": llm_region.get("split_reason", "")
                        })

            # 元の画像を読み込む
            page_dir = os.path.join(TMP_DIR, "analysis_results", file_name, f"page_{page_num}")
            original_image_path = os.path.join(page_dir, "original_page.png")
            original_image = cv2.imread(original_image_path)
            
            if original_image is None:
                st.error("元の画像の読み込みに失敗しました")
                return
            
            # 統合と分割を含む領域処理
            merged_result = process_regions_with_split(analysis_result, llm_analysis, file_name, page_num, original_image)
            
            # 分析結果を保存
            page_manager.add_page_result(page_num, merged_result, llm_analysis)
            
            # 統合後の領域情報を取得
            merged_info = page_manager.get_merged_regions_info(page_num)
            if merged_info:
                # NumPy型を標準のPython型に変換
                merged_info = json.loads(json.dumps(merged_info, default=numpy_to_python))
            
            # 最終的なJSONの保存（統合の有無に関わらず）
            file_dir = os.path.join(TMP_DIR, "analysis_results", file_name)
            json_filename = os.path.join(file_dir, "analysis_results.json")
            
            try:
                # 既存のJSONファイルを読み込むか、新規作成
                if os.path.exists(json_filename):
                    with open(json_filename, "r", encoding="utf-8") as f:
                        all_pages_info = json.load(f)
                else:
                    all_pages_info = {
                        "file_name": file_name,
                        "total_pages": 0,
                        "pages": {}
                    }
                
                # merged_resultから最新の領域情報を取得
                final_regions = merged_result["figures"]
                
                # base_resultを更新（一度だけ）
                base_result = {
                    "total_regions": len(final_regions),
                    "merged_regions": len([r for r in final_regions if r.get("merged_from")]),
                    "split_regions": len([r for r in final_regions if r.get("parent_id")]),
                    "regions": [
                        {
                            "id": str(fig["id"]),
                            "bbox": {
                                "x": int(fig["position"]["x"]),
                                "y": int(fig["position"]["y"]),
                                "width": int(fig["position"]["width"]),
                                "height": int(fig["position"]["height"])
                            },
                            "type": fig.get("type", "未分析"),
                            "description": fig.get("description", ""),
                            "path": str(fig["path"]),
                            "parent_id": fig.get("parent_id"),
                            "split_depth": fig.get("split_depth", 0),
                            "merged_from": fig.get("merged_from", [])
                        }
                        for fig in final_regions
                    ]
                }
                
                # デバッグ用に情報を出力
                logger.info(f"最終的な領域数: {len(final_regions)}")
                logger.info(f"分割された領域数: {len([r for r in final_regions if r.get('parent_id')])}")
                
                # 現在のページの情報を更新
                all_pages_info["pages"][str(page_num)] = base_result
                all_pages_info["total_pages"] = max(
                    all_pages_info["total_pages"],
                    len(all_pages_info["pages"])
                )
                
                # JSONファイルを保存（NumPy型を標準のPython型に変換）
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(all_pages_info, f, ensure_ascii=False, indent=2, 
                             default=numpy_to_python)
                logger.info(f"分析結果をJSONファイルに保存しました: {json_filename}")
                
                # 結果を表示
                st.write("### 領域情報")
                st.json(base_result)
                
            except Exception as e:
                logger.error(f"JSONファイルの保存に失敗しました: {str(e)}")
                st.error(f"結果の保存中にエラーが発生しました: {str(e)}")

def analyze_regions_with_llm(regions, file_name, page_num, overview_path=None):
    """
    LLMを使用して領域を分析する関数
    Args:
        regions: 検出された領域の情報を含むリスト
        file_name: ファイル名
        page_num: ページ番号
        overview_path: 概要画像のパス（オプション）
    Returns:
        dict: 分析結果を含む辞書
    """
    try:
        # 各領域の画像をbase64エンコード
        regions_with_base64 = []
        
        # overview画像を追加（指定されている場合）
        if overview_path and os.path.exists(overview_path):
            with open(overview_path, "rb") as f:
                regions_with_base64.append({
                    "id": "overview",
                    "base64": base64.b64encode(f.read()).decode("utf-8"),
                    "position": None
                })
        
        # 各領域の画像を追加
        for region in regions:
            region_path = os.path.join(region["path"])
            if os.path.exists(region_path):
                with open(region_path, "rb") as f:
                    # 画像を表示
                    st.image(f.read(), caption=f"Region {region['id']}")
                    # ファイルポインタを先頭に戻す
                    f.seek(0)
                    regions_with_base64.append({
                        "id": region["id"],
                        "base64": base64.b64encode(f.read()).decode("utf-8"),
                        "position": region["position"]
                    })

        # LLMに分析を依頼
        prompt = f"""以下の画像は、PDFから検出された領域を示した画像です。最初の画像は全体で、その後は赤い枠線で囲われている各領域の画像が続きます。
赤い枠で囲われた{len(regions)}個のregionについて、以下の情報を分析してください：

1. "type": 領域の種類（図/グラフ/表/テキスト/その他）※複数含まれる場合は「その他」と回答し、"needs_split"はtrueとしてください。
2. "description": 領域の内容の説明
3. "content_types": 領域内に含まれる全ての要素の種類をリストで指定（例：["テキスト", "グラフ"]）
4. "should_merge_with": 1つの図/グラフ/表が複数の領域に分割されてしまっている場合、統合が必要なのでどのregionと統合すべきかを返してください.
5. "needs_split": 以下の場合は必ずtrueを返してください:
   - グラフとテキストが同じ領域に含まれている場合（最優先で分割）
   - 表とテキストが同じ領域に含まれている場合（最優先で分割）
   - 画像とテキストが同じ領域に含まれている場合（最優先で分割）
   - 1つの領域に複数の段落のテキスト、図表が含まれている場合
   - 複数のグラフが1つの領域として検出されている場合
   - 異なる内容や役割を持つ要素が1つの領域に含まれている場合
6. "split_reason": 分割が必要な理由を具体的に記述してください。特に要素の混在（グラフとテキストなど）の場合は明確に記述してください。

重要：適切な分析のためには、領域が適切な粒度で分割されていることが重要です。大きすぎる領域は積極的に分割を提案してください。特に以下の場合は必ず分割が必要です：
- グラフ/表/画像とテキストが同じ領域に混在している（最も優先度の高い分割ケース）
- 複数の図表、グラフが一つの領域に含まれている
- 連続していない複数の段落が一つの領域になっている
- 異なる役割を持つ要素（例：見出しと本文）が一緒になっている

以下のJSON形式で回答してください：
{{
    "regions": [
        {{
            "id": 数値のID,  // "region1"ではなく1を返してください。"region1-1"のように枝番がある場合は1-1と返してください。
            "type": "図/グラフ/表/イメージ/テキスト/その他",
            "description": "内容の説明",
            "content_types": ["テキスト", "グラフ"], // 領域内に含まれる全ての要素の種類
            "should_merge_with": [],
            "needs_split": true/false,
            "split_reason": "XXX"
        }}
    ],
    "merge_suggestions": [
        {{
            "group": [数値のID],  // ["region1", "region2"]ではなく[1, 2]を返してください。"region1-1"のように枝番がある場合は1-1と返してください。
            "reason": "統合を提案する理由"
        }}
    ],
    "split_suggestions": [
        {{
            "region_id": 数値のID,  // "region1"ではなく1を返してください。"region1-1"のように枝番がある場合は1-1と返してください。
            "reason": "分割を提案する/提案しない理由",
            "expected_parts": "予想される分割後の要素数と種類"
        }}
    ]
}}

注意点：
1. IDは文字列（"region1"）ではなくregion以降のみを返してください。"region1-1"のように枝番がある場合は1-1と返してください。
2. 領域の種類は必ず "図"、"グラフ"、"表"、"イメージ"、"テキスト"、"その他" のいずれかを選択してください。その他はできるだけ選択しないでください。
3. グラフ/表/画像とテキストが混在する場合は、必ず分割を提案してください。これは最優先事項です。
4. 分割が必要/不要な場合は、その理由を具体的に説明してください
5. 統合や分割の判断は、領域の内容や配置を考慮して行ってください
6. 文書解析の品質向上のため、曖昧な場合は積極的に分割を提案してください
"""
        # 各領域の画像をLLMに送信
        analysis_result = call_gpt(prompt, image_base64_list=[r["base64"] for r in regions_with_base64])
        
        try:
            result = json.loads(analysis_result)
            
            # IDの形式を検証・変換
            if "regions" in result:
                for region in result["regions"]:
                    # 文字列形式のIDを数値に変換
                    if isinstance(region["id"], str):
                        if region["id"].startswith("region"):
                            region["id"] = region["id"].replace("region", "")
                        else:
                            region["id"] = region["id"]
                    
                    # content_typesの確認と補完
                    if "content_types" not in region:
                        # content_typesが無い場合は、typeから推測
                        region["content_types"] = [region["type"]]
                    
                    # コンテンツタイプに基づいて分割が必要かどうかを再評価
                    content_types = region.get("content_types", [])
                    if len(content_types) > 1:
                        # テキストと他の要素（グラフ/表/画像）が混在する場合
                        if "テキスト" in content_types and any(t in content_types for t in ["グラフ", "表", "イメージ", "図"]):
                            region["needs_split"] = True
                            if not region.get("split_reason"):
                                region["split_reason"] = f"異なる種類の要素（{', '.join(content_types)}）が混在しているため分割が必要"

            # merge_suggestionsのグループIDも変換
            if "merge_suggestions" in result:
                for suggestion in result["merge_suggestions"]:
                    if "group" in suggestion:
                        suggestion["group"] = [
                            g.replace("region", "") if isinstance(g, str) and g.startswith("region")
                            else g if isinstance(g, str)
                            else g
                            for g in suggestion["group"]
                        ]

            # split_suggestionsのregion_idも変換
            if "split_suggestions" in result:
                for suggestion in result["split_suggestions"]:
                    if isinstance(suggestion.get("region_id"), str):
                        if suggestion["region_id"].startswith("region"):
                            suggestion["region_id"] = suggestion["region_id"].replace("region", "")
                        else:
                            suggestion["region_id"] = suggestion["region_id"]

            # 必要なキーの存在確認と形式の検証
            for region in result.get("regions", []):
                if not all(k in region for k in ["id", "type", "description", "needs_split"]):
                    logger.error("領域の分析結果に必要な情報が不足しています")
                    return {"error": "領域の分析結果が不完全です"}
                
                # 型の種類を検証
                if region["type"] not in ["図", "グラフ", "表", "イメージ", "テキスト", "その他"]:
                    region["type"] = "その他"
                
                # needs_splitがTrueの場合、split_reasonの存在を確認
                if region.get("needs_split") and "split_reason" not in region:
                    region["split_reason"] = "分割理由が指定されていません"

            return result

        except json.JSONDecodeError as e:
            logger.error(f"LLMの分析結果をJSONに変換できませんでした: {str(e)}")
            return {"error": "分析結果の解析に失敗しました"}
            
    except Exception as e:
        logger.error(f"領域分析中にエラーが発生しました: {str(e)}")
        return {"error": f"領域分析中にエラーが発生しました: {str(e)}"}

def merge_regions(analysis_result, llm_analysis, file_name, page_num, parent_info=None, parent_offset=None):
    """
    領域のマージを行う関数
    Args:
        analysis_result: 分析結果
        llm_analysis: LLMによる分析結果
        file_name: ファイル名
        page_num: ページ番号
        parent_info: 親領域の情報（サブ領域の場合に使用）
        parent_offset: 親領域の座標オフセット（サブ領域の場合に使用）
    """
    try:
        logger.info(f"merge_regions開始: parent_info={parent_info}, parent_offset={parent_offset}")
        
        # LLM分析結果をIDでマッピング
        llm_info_map = {
            str(region["id"]): region 
            for region in llm_analysis.get("regions", [])
        }

        if not llm_analysis.get("merge_suggestions"):
            # merge_suggestionsがない場合でも、LLM分析情報を付与して返す
            updated_figures = []
            for fig in analysis_result["figures"]:
                updated_fig = fig.copy()
                if str(fig["id"]) in llm_info_map:
                    llm_info = llm_info_map[str(fig["id"])]
                    updated_fig.update({
                        "type": llm_info.get("type", "未分析"),
                        "description": llm_info.get("description", ""),
                        "needs_split": llm_info.get("needs_split", False),
                        "split_reason": llm_info.get("split_reason", "")
                    })
                    if parent_info:
                        updated_fig.update({
                            "parent_id": parent_info["id"],
                            "split_depth": parent_info.get("split_depth", 0) + 1
                        })
                        # 親領域のオフセットを適用
                        if parent_offset:
                            updated_fig["position"]["x"] += parent_offset["x"]
                            updated_fig["position"]["y"] += parent_offset["y"]
                updated_figures.append(updated_fig)
            analysis_result["figures"] = updated_figures
            return analysis_result

        # ファイルとページのディレクトリパスを設定
        file_dir = os.path.join(TMP_DIR, "analysis_results", file_name)
        page_dir = os.path.join(file_dir, f"page_{page_num}")
        
        # サブ領域用のディレクトリを取得（親領域がある場合）
        if parent_info:
            # 既存のサブ領域ディレクトリを使用
            sub_dir = os.path.dirname(analysis_result["figures"][0]["path"])
            logger.info(f"既存のサブ領域ディレクトリを使用: {sub_dir}")
        
        # 元のPDFページ画像を読み込む（高解像度）
        original_image_path = os.path.join(page_dir, "original_page.png")
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            logger.error("元の画像の読み込みに失敗しました")
            return analysis_result
            
        # 結果表示用の画像
        merged_image = original_image.copy()
        
        # 統合された新しい領域のリスト
        merged_figures = []
        used_figures = set()
        
        # 各統合提案を処理
        for suggestion in llm_analysis["merge_suggestions"]:
            group = suggestion["group"]
            if not group:
                continue
                
            # グループ内の全領域の座標を取得
            regions_to_merge = [
                fig for fig in analysis_result["figures"]
                if fig["id"] in group
            ]
            
            if not regions_to_merge:
                continue
                
            # 統合領域の範囲を計算（親領域がある場合は相対座標を考慮）
            min_x = min(r["position"]["x"] for r in regions_to_merge)
            min_y = min(r["position"]["y"] for r in regions_to_merge)
            max_x = max(r["position"]["x"] + r["position"]["width"] for r in regions_to_merge)
            max_y = max(r["position"]["y"] + r["position"]["height"] for r in regions_to_merge)
            
            # 親領域のオフセットを適用
            if parent_offset:
                min_x += parent_offset["x"]
                min_y += parent_offset["y"]
                max_x += parent_offset["x"]
                max_y += parent_offset["y"]
            
            # 新しい統合領域のID（グループの最小IDを使用）
            new_id = min(r["id"] for r in regions_to_merge)
            if parent_info:
                new_id = f"{parent_info['id']}_{new_id}"
            
            logger.info(f"領域をマージ: group={group}, new_id={new_id}")
            
            # 統合領域を赤枠で描画（結果表示用）
            cv2.rectangle(merged_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 4)
            cv2.putText(merged_image, f"merged{new_id}", (min_x, max(30, min_y-15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 7)
            
            # 統合領域を切り出して保存
            margin = 20
            x1 = max(0, min_x - margin)
            y1 = max(0, min_y - margin)
            x2 = min(original_image.shape[1], max_x + margin)
            y2 = min(original_image.shape[0], max_y + margin)
            
            # 元の画像から領域を切り出し
            original_roi = original_image[y1:y2, x1:x2]
            
            # サイズを制限してリサイズ
            resized_roi = resize_with_aspect_ratio(original_roi, max_pixels=1000000)
            
            # マージされた領域の保存パスを設定
            if parent_info:
                merged_path = os.path.join(sub_dir, f"merged_region_{new_id}.png")
            else:
                merged_path = os.path.join(page_dir, f"merged_region_{new_id}.png")
            
            cv2.imwrite(merged_path, resized_roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # 統合領域の情報を追加
            # マージされる領域のdescriptionを取得して結合
            descriptions = []
            types = []
            for region_id in group:
                if str(region_id) in llm_info_map:
                    desc = llm_info_map[str(region_id)].get("description", "")
                    type = llm_info_map[str(region_id)].get("type", "")
                    if desc:
                        descriptions.append(desc)
                    if type:
                        types.append(type)

            merged_description = "\n".join(descriptions) if descriptions else ""
            merged_type = "/".join(types) if types else "未分析"

            merged_region = {
                "id": new_id,
                "path": merged_path,
                "position": {
                    "x": min_x,
                    "y": min_y,
                    "width": max_x - min_x,
                    "height": max_y - min_y
                },
                "merged_from": group,
                "description": merged_description,
                "type": merged_type
            }

            # 親領域の情報がある場合は追加
            if parent_info:
                merged_region.update({
                    "parent_id": parent_info["id"],
                    "split_depth": parent_info.get("split_depth", 0) + 1
                })
            
            merged_figures.append(merged_region)
            used_figures.update(group)
        
        # 統合されなかった領域を追加する際にLLM情報も付与
        for fig in analysis_result["figures"]:
            if fig["id"] not in used_figures:
                updated_fig = fig.copy()
                
                # LLM分析情報を付与
                if str(fig["id"]) in llm_info_map:
                    llm_info = llm_info_map[str(fig["id"])]
                    updated_fig.update({
                        "type": llm_info.get("type", "未分析"),
                        "description": llm_info.get("description", ""),
                        "needs_split": llm_info.get("needs_split", False),
                        "split_reason": llm_info.get("split_reason", "")
                    })

                # 親領域の情報がある場合は追加
                if parent_info:
                    updated_fig.update({
                        "parent_id": parent_info["id"],
                        "split_depth": parent_info.get("split_depth", 0) + 1
                    })
                    # IDを親領域のIDと組み合わせる
                    updated_fig["id"] = f"{parent_info['id']}_{fig['id']}"
                    
                # 親領域のオフセットを適用
                if parent_offset:
                    updated_fig["position"]["x"] += parent_offset["x"]
                    updated_fig["position"]["y"] += parent_offset["y"]

                # 元の画像から領域を切り出し
                pos = updated_fig["position"]
                margin = 20
                x1 = max(0, pos["x"] - margin)
                y1 = max(0, pos["y"] - margin)
                x2 = min(original_image.shape[1], pos["x"] + pos["width"] + margin)
                y2 = min(original_image.shape[0], pos["y"] + pos["height"] + margin)
                
                original_roi = original_image[y1:y2, x1:x2]
                resized_roi = resize_with_aspect_ratio(original_roi, max_pixels=1000000)
                
                # 保存パスを設定
                if parent_info:
                    new_path = os.path.join(sub_dir, f"region_{updated_fig['id']}.png")
                else:
                    new_path = os.path.join(page_dir, f"region_{fig['id']}.png")
                
                cv2.imwrite(new_path, resized_roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                updated_fig["path"] = new_path
                merged_figures.append(updated_fig)
                
                # 非統合領域を緑色で描画（結果表示用）
                cv2.rectangle(merged_image, 
                            (pos["x"], pos["y"]), 
                            (pos["x"] + pos["width"], pos["y"] + pos["height"]), 
                            (0, 255, 0), 4)
                cv2.putText(merged_image, f"region{updated_fig['id']}", 
                           (pos["x"], max(30, pos["y"]-15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 7)
        
        # 統合後の画像を保存（結果表示用）
        if parent_info:
            merged_image_path = os.path.join(sub_dir, "merged_regions.png")
        else:
            merged_image_path = os.path.join(page_dir, "merged_regions.png")
        
        cv2.imwrite(merged_image_path, merged_image)
        logger.info(f"マージ結果を保存: {merged_image_path}")
        
        # 結果を返す
        return {
            "figures": merged_figures,
            "result_image": merged_image_path
        }
        
    except Exception as e:
        logger.error(f"領域の統合中にエラーが発生しました: {str(e)}")
        return analysis_result

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

def analyze_image_content_with_connected_components(image_path, file_name, page_num, is_subsplit=False, recursion_depth=0):
    """
    連結成分解析を使用して画像から大きな領域を抽出する関数
    重複や近接している領域は統合します。
    Args:
        image_path: 分析対象の画像パス
        file_name: 分析対象のファイル名
        page_num: ページ番号
        is_subsplit: 再分割処理かどうか（Trueの場合はより細かい分割を行う）
        recursion_depth: 現在の再帰深度（深くなるにつれて徐々に細かくなる）
    Returns:
        dict: 抽出結果を含む辞書
    """
    try:
        # ファイルごとのディレクトリを作成
        file_dir = os.path.join(TMP_DIR, "analysis_results", file_name)
        os.makedirs(file_dir, exist_ok=True)
        
        # ページごとのディレクトリを作成
        page_dir = os.path.join(file_dir, f"page_{page_num}")
        os.makedirs(page_dir, exist_ok=True)

        # 元の画像を適切な場所にコピー
        target_image_path = os.path.join(page_dir, "source_page.jpg")
        shutil.copy2(image_path, target_image_path)

        # 画像読み込み
        original_img = cv2.imread(target_image_path)  # カラー画像として読み込み
        if original_img is None:
            return {"error": "画像の読み込みに失敗しました"}

        # 元の画像を保存（高画質）
        original_page_path = os.path.join(page_dir, "original_page.png")
        cv2.imwrite(original_page_path, original_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # グレースケールに変換
        img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img.shape

        # 再分割時は深さに応じてパラメータを段階的に調整
        if is_subsplit:
            # 段階的に細かくなるように調整（1段階目は控えめな分割）
            depth_factor = min(1.0, 0.4 + (recursion_depth * 0.3))  # 0.4→0.7→1.0と徐々に細かく
            
            # 最小領域サイズ（再帰深度が深くなるに従って小さくなる）
            min_area = (img_width * img_height) * (0.002 - (recursion_depth * 0.0005))
            min_area = max(min_area, (img_width * img_height) * 0.0005)  # 下限値を設定
            
            # 最大領域サイズも深さに応じて段階的に調整
            max_area = (img_width * img_height) * (0.8 - (recursion_depth * 0.2))
            max_area = max(max_area, (img_width * img_height) * 0.3)  # 下限値を設定
            
            # ブラーやカーネルサイズも深さに応じて段階的に調整
            # medianBlurのカーネルサイズは奇数でなければならない
            blur_size = 5 - recursion_depth * 2  # 5→3→1と奇数のみで調整
            blur_size = max(3, blur_size)  # 最小値は3に制限
            
            kernel_size = max(3, min(img_width, img_height) // (50 + (recursion_depth * 25)))
            # カーネルサイズも奇数に調整
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            
            dilation_iterations = max(1, 3 - recursion_depth)  # 3→2→1と徐々に少なく
            
            logger.info(f"再帰深度{recursion_depth}での分割パラメータ: min_area={min_area}, max_area={max_area}")
            logger.info(f"blur_size={blur_size}, kernel_size={kernel_size}, dilation_iterations={dilation_iterations}")
        else:
            min_area = (img_width * img_height) * 0.002  # 画像全体の0.2%に下げる
            max_area = (img_width * img_height) * 0.9   # 画像全体の90%
            blur_size = 5
            kernel_size = min(img_width, img_height) // 50  # より小さいカーネルを使用
            # カーネルサイズを奇数に調整
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            dilation_iterations = 2  # 膨張処理を弱める

        # カーネルサイズの制限（必ず奇数になるように調整）
        kernel_size = max(3, min(kernel_size, 49))  # 最小3、最大49に制限（奇数に調整）

        # メディアンブラーでノイズを軽減
        blurred = cv2.medianBlur(img, blur_size)

        # 二値化（Otsuの方法で自動的にしきい値を決定）
        _, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # モルフォロジー演算用のカーネル
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # 膨張処理で領域を結合
        dilated = cv2.dilate(bin_img, kernel, iterations=dilation_iterations)
        
        # 収縮処理で形状を整える
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # 連結成分解析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)

        # 結果の可視化用の画像
        color_img = original_img.copy()
        
        # 重なり領域をチェックする関数
        def check_overlap(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            
            # 各矩形の端点を計算
            left1, right1 = x1, x1 + w1
            top1, bottom1 = y1, y1 + h1
            left2, right2 = x2, x2 + w2
            top2, bottom2 = y2, y2 + h2
            
            # 重なりがない場合
            if right1 < left2 or right2 < left1 or bottom1 < top2 or bottom2 < top1:
                return False, 0.0
            
            # 重なり領域の計算
            overlap_width = min(right1, right2) - max(left1, left2)
            overlap_height = min(bottom1, bottom2) - max(top1, top2)
            overlap_area = overlap_width * overlap_height
            
            # 小さい方の矩形に対する重なりの割合を計算
            area1 = w1 * h1
            area2 = w2 * h2
            min_area = min(area1, area2)
            overlap_ratio = overlap_area / min_area
            
            return True, overlap_ratio

        # 近接または重なっている領域を統合する関数
        def should_merge_regions(rect1, rect2):
            # 重なりチェック
            has_overlap, overlap_ratio = check_overlap(rect1, rect2)
            
            # 再帰深度に応じて重なりの閾値を調整
            # 深い再帰では、より厳しい（高い）閾値を使用
            overlap_threshold = 0.01  # デフォルト値
            if is_subsplit:
                # 再帰深度に応じて閾値を高くする（より統合しにくくする）
                overlap_threshold = 0.1 + (recursion_depth * 0.15)
                
            if has_overlap and overlap_ratio > overlap_threshold:  
                return True
            
            # 重なりがない場合は距離をチェック
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2
            
            # 中心点の計算
            center1 = (x1 + w1/2, y1 + h1/2)
            center2 = (x2 + w2/2, y2 + h2/2)
            
            # 各矩形の対角線の長さを計算
            diag1 = np.sqrt(w1**2 + h1**2)
            diag2 = np.sqrt(w2**2 + h2**2)
            
            # 中心点間の距離を計算
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            
            # 再分割時は距離の閾値を深さに応じて調整
            base_factor = 0.3  # 基準係数
            if is_subsplit:
                # 再帰深度に応じて係数を小さくする（より統合しにくくする）
                depth_factor = max(0.1, base_factor - (recursion_depth * 0.1))  # 0.3→0.2→0.1と徐々に減少
            else:
                depth_factor = base_factor
                
            distance_threshold = min(diag1, diag2) * depth_factor
            return distance < distance_threshold

        # 領域の情報を一時保存
        temp_regions = []
        
        # 領域を処理（0番は背景なので除外）
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # サイズと面積でフィルタリング
            min_size = 20 if is_subsplit else 50  # 再分割時は小さい領域も許容
            if min_area < area < max_area and w > min_size and h > min_size:
                temp_regions.append([x, y, w, h])

        # 領域の統合を繰り返し実行（変更がなくなるまで）
        while True:
            merged_regions = []
            used_indices = set()
            changes_made = False

            for i, region1 in enumerate(temp_regions):
                if i in used_indices:
                    continue

                current_region = list(region1)
                merged_indices = {i}
                
                # 他の領域との統合をチェック
                for j, region2 in enumerate(temp_regions):
                    if j in used_indices or i == j:
                        continue

                    if should_merge_regions(current_region, region2):
                        # 領域を統合
                        x1, y1, w1, h1 = current_region
                        x2, y2, w2, h2 = region2
                        
                        # 新しい外接矩形を計算
                        new_x = min(x1, x2)
                        new_y = min(y1, y2)
                        new_w = max(x1 + w1, x2 + w2) - new_x
                        new_h = max(y1 + h1, y2 + h2) - new_y
                        
                        current_region = [new_x, new_y, new_w, new_h]
                        merged_indices.add(j)
                        changes_made = True

                # 使用したインデックスを記録
                used_indices.update(merged_indices)
                merged_regions.append(current_region)

            # 変更がなければループを終了
            if not changes_made:
                break

            temp_regions = merged_regions

        # 統合された領域を処理
        areas = []
        for i, (x, y, w, h) in enumerate(temp_regions):
            # 余白を追加
            margin = 20  # マージンを20ピクセルに減らす
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img_width, x + w + margin)
            y2 = min(img_height, y + h + margin)
            
            # 領域を可視化（太い線で描画）
            cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 4)
            
            # 領域名を表示（サイズを1.5に、太さを3に変更）
            area_name = f"region{i+1}"
            cv2.putText(color_img, area_name, (x, max(30, y-15)), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 7)
            
            # 元の画像から領域を切り出し
            original_roi = original_img[y1:y2, x1:x2]
            
            # サイズを制限してリサイズ
            resized_roi = resize_with_aspect_ratio(original_roi, max_pixels=1000000)  # 1MPに制限
            
            area_path = os.path.join(page_dir, f"region_{i+1}.png")
            cv2.imwrite(area_path, resized_roi, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            areas.append({
                "id": i+1,
                "path": area_path,
                "position": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            })

        # 検出結果の可視化画像を保存
        result_image_path = os.path.join(page_dir, "detected_regions.png")
        cv2.imwrite(result_image_path, color_img)
        
        return {
            "figures": areas,  # すべての領域を figures として返す
            "tables": [],     # テーブルの個別検出は行わない
            "result_image": result_image_path
        }
        
    except Exception as e:
        logger.error(f"画像分析中にエラーが発生しました: {str(e)}")
        return {"error": f"画像分析中にエラーが発生しました: {str(e)}"}

def split_region_using_existing_process(region_info, original_image, file_name, page_num, recursion_depth=0, max_depth=2):
    """
    既存の連結成分分析を使用して領域をさらに分割する関数
    
    Args:
        region_info: 分割対象の領域情報
        original_image: 元の画像
        file_name: ファイル名
        page_num: ページ番号
        recursion_depth: 現在の再帰の深さ
        max_depth: 最大再帰深さ
    """
    try:
        if recursion_depth >= max_depth:
            logger.info(f"最大再帰深さ({max_depth})に達しました")
            return [region_info]

        logger.info(f"領域 {region_info['id']} の分割処理を開始（再帰深度: {recursion_depth}）")

        # 領域を切り出し
        x, y = region_info["position"]["x"], region_info["position"]["y"]
        w, h = region_info["position"]["width"], region_info["position"]["height"]
        
        # 余白を追加
        margin = 10  # 余白を小さくして分割精度を向上
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(original_image.shape[1], x + w + margin)
        y2 = min(original_image.shape[0], y + h + margin)
        
        roi = original_image[y1:y2, x1:x2].copy()
        
        # 分割モードの判定（テキストとグラフ/表/画像が混在する場合など）
        mixed_content = False
        content_types = []
        
        # region_infoからcontent_typesフィールドがあれば取得
        if "content_types" in region_info:
            content_types = region_info["content_types"]
            if "テキスト" in content_types and any(t in content_types for t in ["グラフ", "表", "イメージ", "図"]):
                mixed_content = True
                logger.info(f"領域 {region_info['id']} はテキストと他の要素が混在しています: {content_types}")
        
        # 切り出した画像を一時ファイルとして保存
        temp_roi_path = os.path.join(TMP_DIR, f"temp_roi_{region_info['id']}.png")
        cv2.imwrite(temp_roi_path, roi)
        
        # 一時ファイルをBytesIOに変換（既存の処理が期待する形式に合わせる）
        with open(temp_roi_path, 'rb') as f:
            roi_bytes = io.BytesIO(f.read())
        
        # 混在コンテンツの場合は特別な分割処理を適用
        if mixed_content:
            logger.info("混在コンテンツ用の強化分割処理を適用します")
            
            # グレースケールに変換
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 二値化（テキストと非テキスト領域を分離しやすくする）
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # モルフォロジー演算でノイズを除去し、テキスト領域を強調
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # テキスト特有の特徴を検出するため、水平方向に繋がった成分を見つける
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(opening, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # テキスト領域を推定（水平方向の線が多い領域）
            text_mask = cv2.dilate(detected_lines, kernel, iterations=3)
            
            # 連結成分の検出（テキスト以外の大きな領域を見つける）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
            
            # 面積によるフィルタリング（大きな領域はグラフや表、画像の可能性が高い）
            min_size = 500  # 最小サイズのしきい値
            non_text_regions = []
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x_cc = stats[i, cv2.CC_STAT_LEFT]
                y_cc = stats[i, cv2.CC_STAT_TOP]
                w_cc = stats[i, cv2.CC_STAT_WIDTH]
                h_cc = stats[i, cv2.CC_STAT_HEIGHT]
                
                # 一定以上の大きさの領域を非テキスト領域候補とする
                if area > min_size and w_cc > 50 and h_cc > 50:
                    # テキスト特徴量が少ない領域を非テキスト領域とする
                    component_mask = (labels == i).astype(np.uint8) * 255
                    text_overlap = cv2.bitwise_and(component_mask, text_mask)
                    text_ratio = np.sum(text_overlap) / np.sum(component_mask)
                    
                    if text_ratio < 0.3:  # テキスト特徴が30%未満なら非テキスト領域
                        non_text_regions.append((x_cc, y_cc, w_cc, h_cc))
            
            # 非テキスト領域（グラフ/表/画像）とテキスト領域を分離
            regions = []
            
            # 非テキスト領域をまず追加
            for i, (x_nt, y_nt, w_nt, h_nt) in enumerate(non_text_regions):
                regions.append({
                    "id": f"{region_info['id']}_nt{i+1}",
                    "position": {
                        "x": x_nt,
                        "y": y_nt,
                        "width": w_nt,
                        "height": h_nt
                    },
                    "parent_id": str(region_info["id"]),
                    "split_depth": recursion_depth + 1,
                    "type": "その他"  # LLM分析で詳細を判断
                })
            
            # 非テキスト領域がない場合や検出に失敗した場合は、通常の分割処理を使用
            if len(regions) == 0:
                logger.info("混在コンテンツの特殊分割が失敗しました。通常の分割処理を使用します。")
                use_default_process = True
            else:
                # テキスト領域を検出（非テキスト領域以外の部分）
                text_regions_mask = np.ones_like(gray) * 255
                for x_nt, y_nt, w_nt, h_nt in non_text_regions:
                    text_regions_mask[y_nt:y_nt+h_nt, x_nt:x_nt+w_nt] = 0
                
                # テキスト領域のみのマスクを作成
                text_only = cv2.bitwise_and(binary, text_regions_mask)
                
                # テキスト領域の連結成分を検出
                num_text_labels, text_labels, text_stats, _ = cv2.connectedComponentsWithStats(text_only)
                
                # テキスト領域を追加
                for i in range(1, num_text_labels):
                    area = text_stats[i, cv2.CC_STAT_AREA]
                    if area > 200:  # 小さすぎる領域は無視
                        x_t = text_stats[i, cv2.CC_STAT_LEFT]
                        y_t = text_stats[i, cv2.CC_STAT_TOP]
                        w_t = text_stats[i, cv2.CC_STAT_WIDTH]
                        h_t = text_stats[i, cv2.CC_STAT_HEIGHT]
                        
                        regions.append({
                            "id": f"{region_info['id']}_t{i}",
                            "position": {
                                "x": x_t,
                                "y": y_t,
                                "width": w_t,
                                "height": h_t
                            },
                            "parent_id": str(region_info["id"]),
                            "split_depth": recursion_depth + 1,
                            "type": "テキスト"
                        })
                
                # それでもテキスト領域が検出できなければ、全体をテキスト領域として追加
                if len(regions) == len(non_text_regions):
                    regions.append({
                        "id": f"{region_info['id']}_text",
                        "position": {
                            "x": 0,
                            "y": 0,
                            "width": roi.shape[1],
                            "height": roi.shape[0]
                        },
                        "parent_id": str(region_info["id"]),
                        "split_depth": recursion_depth + 1,
                        "type": "テキスト"
                    })
                
                # 分割結果を視覚化
                vis_img = roi.copy()
                for r in regions:
                    rx = r["position"]["x"]
                    ry = r["position"]["y"]
                    rw = r["position"]["width"]
                    rh = r["position"]["height"]
                    color = (0, 255, 0) if r["type"] == "テキスト" else (0, 0, 255)
                    cv2.rectangle(vis_img, (rx, ry), (rx+rw, ry+rh), color, 2)
                
                # 可視化画像を保存
                vis_path = os.path.join(TMP_DIR, f"mixed_split_{region_info['id']}.png")
                cv2.imwrite(vis_path, vis_img)
                
                # 分割結果を返す準備
                logger.info(f"混在コンテンツの分割により {len(regions)} 個の領域を検出")
                
                # パスを設定
                for i, r in enumerate(regions):
                    sub_roi = roi[r["position"]["y"]:r["position"]["y"]+r["position"]["height"], 
                                r["position"]["x"]:r["position"]["x"]+r["position"]["width"]]
                    sub_path = os.path.join(TMP_DIR, f"sub_region_{r['id']}.png")
                    cv2.imwrite(sub_path, sub_roi)
                    r["path"] = sub_path
                
                # 最後に親領域の座標を加算して返す
                for region in regions:
                    region.update({
                        "position": {
                            "x": region["position"]["x"] + x1,
                            "y": region["position"]["y"] + y1,
                            "width": region["position"]["width"],
                            "height": region["position"]["height"]
                        }
                    })
                
                return regions
        
        # 特殊な分割処理を適用しない場合、または特殊分割が失敗した場合は標準の分割処理を使用
        # 連結成分分析を使用して分割（再帰深度を渡す）
        sub_regions_result = analyze_image_content_with_connected_components(
            temp_roi_path,
            f"{file_name}_sub_{region_info['id']}",
            f"{page_num}_{region_info['id']}",
            is_subsplit=True,  # 分割モードを有効化
            recursion_depth=recursion_depth  # 現在の再帰深度を渡す
        )
        
        # 分割に失敗した場合は元の領域を返す
        if "error" in sub_regions_result or len(sub_regions_result["figures"]) <= 1:
            logger.info(f"領域 {region_info['id']} は分割できませんでした")
            return [region_info]

        # サブ領域の座標は相対座標のまま保持
        logger.info(f"検出されたサブ領域の数: {len(sub_regions_result['figures'])}")
        
        # サブ領域に一意のIDを割り当て（親のIDを含む形式）
        parent_id = str(region_info["id"])
        for i, region in enumerate(sub_regions_result["figures"]):
            # 子領域のIDを親領域のIDと関連付けて一意にする
            region["id"] = f"{parent_id}_{i+1}"
            # 親領域のIDを記録
            region["parent_id"] = parent_id
            # 分割深度を記録
            region["split_depth"] = recursion_depth + 1
        
        # サブ領域のLLM分析を実行（相対座標のまま）
        sub_regions_for_llm = [
            {
                "id": str(region["id"]),
                "position": {
                    "x": region["position"]["x"],  # 相対座標のまま
                    "y": region["position"]["y"],
                    "width": region["position"]["width"],
                    "height": region["position"]["height"]
                },
                "path": region["path"]
            }
            for region in sub_regions_result["figures"]
        ]

        # デバッグ用：LLM分析前の領域情報を出力
        logger.info("LLM分析前の領域情報:")
        logger.info(f"sub_regions_for_llm: {json.dumps(sub_regions_for_llm, indent=2)}")

        # LLM分析を実行
        sub_llm_analysis = analyze_regions_with_llm(
            sub_regions_for_llm, 
            file_name, 
            page_num,
            overview_path=sub_regions_result["result_image"]
        )

        if isinstance(sub_llm_analysis, str):
            try:
                sub_llm_analysis = json.loads(sub_llm_analysis)
            except json.JSONDecodeError:
                logger.error("サブ領域のLLM分析結果のJSONパースに失敗")
                sub_llm_analysis = {"error": "分析結果の解析に失敗しました"}

        # デバッグ用：LLM分析結果を出力
        logger.info("LLM分析結果:")
        logger.info(f"sub_llm_analysis: {json.dumps(sub_llm_analysis, indent=2, ensure_ascii=False)}")

        # 分析結果がエラーでない場合
        if "error" not in sub_llm_analysis and "regions" in sub_llm_analysis:
            # サブ領域の分析結果を統合
            analyzed_regions = {r["id"]: r for r in sub_llm_analysis["regions"]}
            
            # 分析結果を各領域に適用
            for region in sub_regions_result["figures"]:
                region_id = str(region["id"])
                if region_id in analyzed_regions:
                    # 分析結果から必要な情報を取得
                    analysis = analyzed_regions[region_id]
                    
                    # typeプロパティを確実に設定（見つからない場合はデフォルト値を使用）
                    region["type"] = analysis.get("type", "unknown")
                    region["content"] = analysis.get("content", "")
                    region["needs_split"] = analysis.get("needs_split", False)
                    
                    # content_typesがあれば設定
                    if "content_types" in analysis:
                        region["content_types"] = analysis["content_types"]
                else:
                    # 分析結果がない場合はデフォルト値を設定
                    region["type"] = "unknown"
                    region["content"] = ""
                    region["needs_split"] = False
                    region["content_types"] = [region["type"]]
            
            # サブ領域に対してマージを実行（親領域の情報を渡す）
            logger.info("サブ領域のマージ処理を開始")
            
            # 親領域のオフセット情報を追加
            parent_offset = {
                "x": x1,
                "y": y1
            }
            
            merged_result = merge_regions(
                sub_regions_result,
                sub_llm_analysis,
                file_name,
                page_num,
                parent_info=region_info,
                parent_offset=parent_offset  # オフセット情報を追加
            )

            if "error" not in merged_result:
                logger.info(f"サブ領域のマージ完了: {len(merged_result['figures'])} 領域")
                # マージ結果にも親のIDと深さの情報を設定
                for merged_region in merged_result["figures"]:
                    if "parent_id" not in merged_region:
                        merged_region["parent_id"] = parent_id
                    if "split_depth" not in merged_region:
                        merged_region["split_depth"] = recursion_depth + 1
                return merged_result["figures"]
        
        # エラーが発生した場合やマージ提案がない場合は、元の分割結果を返す
        logger.info("マージ処理をスキップし、元の分割結果を返します")
        
        # 最後に親領域の座標を加算して返す
        for region in sub_regions_result["figures"]:
            # typeプロパティが設定されていない場合はデフォルト値を設定
            if "type" not in region:
                region["type"] = "unknown"
            
            region.update({
                "position": {
                    "x": region["position"]["x"] + x1,
                    "y": region["position"]["y"] + y1,
                    "width": region["position"]["width"],
                    "height": region["position"]["height"]
                }
            })
        return sub_regions_result["figures"]
        
    except Exception as e:
        logger.error(f"領域分割処理でエラーが発生: {str(e)}")
        traceback.print_exc()  # スタックトレースを出力
        return [region_info]

def process_regions_with_split(analysis_result, llm_analysis, file_name, page_num, original_image):
    """
    統合と分割を含む領域処理のメイン関数
    """
    try:
        # 1. まず領域の統合を実行
        merged_result = merge_regions(analysis_result, llm_analysis, file_name, page_num)
        
        # 最大分割深度を設定（少し控えめに設定）
        max_split_depth = 2
        
        # 分割優先度の高い領域（テキストとグラフ/表/画像が混在する領域）を識別
        priority_regions = []
        normal_regions = []
        
        for region in merged_result["figures"]:
            region_analysis = next(
                (r for r in llm_analysis["regions"] if r["id"] == str(region["id"])),
                None
            )
            
            if region_analysis and region_analysis.get("needs_split", False):
                # content_typesが存在し、テキストと他の要素が混在している場合は優先的に分割
                mixed_content = False
                if "content_types" in region_analysis:
                    content_types = region_analysis["content_types"]
                    if "テキスト" in content_types and any(t in content_types for t in ["グラフ", "表", "イメージ", "図"]):
                        mixed_content = True
                
                # 分割理由にテキストとグラフ/表/画像の混在が明示されている場合も優先
                split_reason = region_analysis.get("split_reason", "").lower()
                mixed_keywords = ["テキスト", "グラフ", "表", "混在", "混合"]
                reason_indicates_mixed = any(keyword in split_reason for keyword in mixed_keywords)
                
                if mixed_content or reason_indicates_mixed:
                    # 優先度の高い領域として登録
                    region["is_mixed_content"] = True
                    region["content_types"] = region_analysis.get("content_types", [region_analysis.get("type", "unknown")])
                    priority_regions.append(region)
                else:
                    # 通常の分割対象領域
                    region["is_mixed_content"] = False
                    normal_regions.append(region)
            else:
                # 分割が不要な領域
                if region_analysis and "type" in region_analysis:
                    region["type"] = region_analysis["type"]
                elif "type" not in region:
                    region["type"] = "unknown"
                
                normal_regions.append(region)
        
        # 段階的に分割を実行（優先領域からスタート）
        logger.info(f"優先的に分割する混在コンテンツ領域数: {len(priority_regions)}")
        
        final_regions = []
        
        # 1. まず優先的に分割する領域を処理
        for region in priority_regions:
            logger.info(f"優先領域 {region['id']} の分割を開始")
            
            # 領域の種類を格納
            if "content_types" not in region and "content_types" in llm_analysis["regions"][0]:
                region_analysis = next(
                    (r for r in llm_analysis["regions"] if r["id"] == str(region["id"])),
                    None
                )
                if region_analysis and "content_types" in region_analysis:
                    region["content_types"] = region_analysis["content_types"]
            
            # 強化された分割処理で領域を分割
            split_regions = split_region_using_existing_process(
                region,
                original_image,
                file_name,
                page_num,
                recursion_depth=0,
                max_depth=max_split_depth
            )
            
            # 分割結果を最終リストに追加
            final_regions.extend(split_regions)
        
        # 2. 次に通常の分割処理を段階的に実行
        for current_depth in range(max_split_depth + 1):
            logger.info(f"通常分割処理: 深度 {current_depth} を開始")
            depth_regions = []
            
            # 現在の深度の領域を処理
            for region in normal_regions:
                # 既に指定された深度まで分割済みの領域はスキップ
                if region.get("split_depth", 0) > current_depth:
                    final_regions.append(region)
                    continue
                
                region_analysis = next(
                    (r for r in llm_analysis["regions"] if r["id"] == str(region["id"])),
                    None
                )
                
                # 分割が必要な領域を処理（現在の深度のものだけ）
                if region_analysis and region_analysis.get("needs_split", False):
                    logger.info(f"領域 {region['id']} の分割を開始（深度: {current_depth}）")
                    
                    # 深度を指定して領域を分割
                    split_regions = split_region_using_existing_process(
                        region,
                        original_image,
                        file_name,
                        page_num,
                        recursion_depth=current_depth,
                        max_depth=current_depth + 1  # 1段階だけ分割
                    )
                    depth_regions.extend(split_regions)
                else:
                    # 分割しない場合もtypeプロパティが確実に設定されるようにする
                    if region_analysis and "type" in region_analysis:
                        region["type"] = region_analysis["type"]
                    elif "type" not in region:
                        region["type"] = "unknown"
                    
                    depth_regions.append(region)
            
            # 各深度で処理された領域を次のサイクルに渡す
            normal_regions = depth_regions
            
            # 最終サイクルでfinal_regionsに追加
            if current_depth == max_split_depth:
                final_regions.extend(depth_regions)
            
            # 現在の深度の分割結果に対する領域分析を実行
            if current_depth < max_split_depth and depth_regions:
                regions_for_llm = [
                    {
                        "id": str(region["id"]),
                        "position": region["position"],
                        "path": region.get("path", "")
                    } 
                    for region in depth_regions
                    if "path" in region
                ]
                
                if regions_for_llm:
                    # 分割された領域のGPT分析を実行（分割が必要かを判断）
                    new_llm_analysis = analyze_regions_with_llm(
                        regions_for_llm,
                        file_name,
                        page_num
                    )
                    
                    # 新しい分析結果を使用
                    if not isinstance(new_llm_analysis, dict):
                        try:
                            new_llm_analysis = json.loads(new_llm_analysis)
                        except:
                            logger.error("LLM分析結果のJSONパースに失敗")
                            continue
                    
                    if isinstance(new_llm_analysis, dict) and "regions" in new_llm_analysis:
                        # 分析結果をdictに変換して効率的に参照できるようにする
                        analyzed_regions = {r["id"]: r for r in new_llm_analysis["regions"]}
                        
                        # 分析結果を各領域に適用
                        for region in depth_regions:
                            region_id = str(region["id"])
                            if region_id in analyzed_regions:
                                analysis = analyzed_regions[region_id]
                                # typeプロパティを確実に設定
                                region["type"] = analysis.get("type", region.get("type", "unknown"))
                                region["content"] = analysis.get("content", region.get("content", ""))
                                region["needs_split"] = analysis.get("needs_split", False)
                                if "content_types" in analysis:
                                    region["content_types"] = analysis["content_types"]
                        
                        # 次の分割サイクルのために更新された分析結果を使用
                        llm_analysis = new_llm_analysis
        
        # 最終的な分析結果のチェック
        for region in final_regions:
            if "type" not in region or not region["type"] or region["type"] == "unknown":
                logger.warning(f"領域 {region['id']} のtypeが未設定または不明です。デフォルト値を設定します。")
                region["type"] = "text"  # デフォルトタイプをtextに設定
        
        # 最終結果を設定
        merged_result["figures"] = final_regions
        
        # 結果の可視化
        visualize_final_regions(original_image, final_regions, file_name, page_num)
        
        return merged_result
        
    except Exception as e:
        logger.error(f"領域処理でエラーが発生: {str(e)}")
        traceback.print_exc()  # スタックトレースを出力
        return analysis_result

def visualize_final_regions(original_image, final_regions, file_name, page_num):
    """
    最終的な領域を可視化する関数
    """
    try:
        # 可視化用の画像をコピー
        visualization_image = original_image.copy()
        
        # 深さに応じた色を定義（深さ0から3まで）
        colors = [
            (0, 255, 0),    # 緑（深さ0）
            (255, 0, 0),    # 赤（深さ1）
            (0, 0, 255),    # 青（深さ2）
            (255, 255, 0)   # 黄（深さ3以上）
        ]
        
        logger.info(f"領域の可視化を開始: 総領域数={len(final_regions)}")
        
        # 各領域を描画
        for region in final_regions:
            # 領域の深さを取得（デフォルトは0）
            depth = region.get("split_depth", 0)
            color = colors[min(depth, len(colors) - 1)]
            
            # 領域の座標を取得
            pos = region["position"]
            x, y = pos["x"], pos["y"]
            w, h = pos["width"], pos["height"]
            
            # マージされた領域かどうかを確認
            is_merged = bool(region.get("merged_from"))
            line_thickness = 6 if is_merged else 4
            
            # 矩形を描画
            cv2.rectangle(visualization_image, 
                        (x, y), 
                        (x + w, y + h), 
                        color, line_thickness)
            
            # 領域IDを表示
            region_id = str(region["id"])
            prefix = "merged" if is_merged else "region"
            cv2.putText(visualization_image, 
                       f"{prefix}{region_id}", 
                       (x, max(30, y-15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       3, color, 5)
            
            # 親領域がある場合は、その情報も表示
            if "parent_id" in region:
                cv2.putText(visualization_image,
                           f"(from {region['parent_id']})",
                           (x, max(60, y-45)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           2, color, 3)
            
            # マージ情報を表示
            if is_merged:
                merged_from = region["merged_from"]
                cv2.putText(visualization_image,
                           f"merged: {merged_from}",
                           (x, y + h + 30),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           2, color, 3)
            
            logger.info(f"領域を描画: id={region_id}, depth={depth}, merged={is_merged}")
        
        # 結果画像を保存
        output_dir = os.path.join(TMP_DIR, "analysis_results", file_name, f"page_{page_num}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_regions.png")
        cv2.imwrite(output_path, visualization_image)
        
        logger.info(f"最終的な領域の可視化結果を保存しました: {output_path}")
        
    except Exception as e:
        logger.error(f"領域の可視化中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()