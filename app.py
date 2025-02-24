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
# 環境変数からAPIキーを取得
openai.api_key = os.getenv("OPENAI_API_KEY")
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
def call_gpt(prompt, image_base64_list=None, max_tokens=1500, temperature=0.0, model="chatgpt-4o-latest"):
    """
    LLMを呼び出すユーティリティ関数
    Args:
        prompt: プロンプトテキスト
        image_base64_list: Base64エンコードされた画像のリスト（複数画像対応）
        max_tokens: 最大トークン数
        temperature: 温度パラメータ
        model: 使用するモデル
    """
    messages = [
        {"role": "system", "content": "あなたは与えられた画像から俯瞰的な目線で正確に文書構造を読み取り、JSON形式で返してください。"}
    ]
    
    if image_base64_list:
        # 画像が1つ以上ある場合
        logger.info(f"image_base64_list:{len(image_base64_list)}")
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

    # messagesをテキストファイルに出力
    with open("messages.txt", "w", encoding="utf-8") as f:
        f.write(str(messages))
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    # st.write(response.choices[0].message.content)
    return response.choices[0].message.content

##############################
# PDFをページ単位で画像変換 #
##############################
def pdf_to_images(pdf_file, dpi=1000):
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
        return []

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
    e_page = total_pages - 1
    
    s_page = max(0, min(s_page, total_pages-1))
    e_page = max(0, min(e_page, total_pages-1))
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
                
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        analysis_results = json.load(f)
                    
                    # s_pageからe_pageまでの画像を取得
                    for page_num in range(s_page, e_page + 1):
                        if "pages" in analysis_results:
                            for page_info in analysis_results["pages"][str(page_num)]["regions"]:
                                path = page_info["path"]
                                type = page_info["type"]
                                # typeに「図」「イメージ」「表」「グラフ」が含まれているかチェック
                                if "図" in type or "イメージ" in type or "表" in type or "グラフ" in type:
                                    if os.path.exists(path):
                                        with open(path, "rb") as img_file:
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
            create_section_summary(subsec, pages_images, total_pages, file_name,
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
        create_section_summary(section, pages_images, total_pages, file_name, 1, summary_depth, content_depth)

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

    # ファイルアップロード
    uploaded_file = st.file_uploader("PDFファイルをアップロード", type=['pdf'])

    if uploaded_file is not None:
        try:
            # ファイル名を取得（拡張子を除く）
            file_name = os.path.splitext(uploaded_file.name)[0]
            
            # PDFを画像に変換
            pages_images = pdf_to_images(uploaded_file)
            if not pages_images:
                st.error("PDFの処理に失敗しました。")
                return

            try:
                # タブを作成
                tab1, tab2, tab3 = st.tabs(["文書構造分析", "画像分析", "分析結果サマリー"])

                with tab1:
                    st.header("文書構造分析")
                    # 既存の文書構造分析の処理
                    chunk_size = st.sidebar.slider("チャンクサイズ", min_value=1, max_value=20, value=10)
                    summary_depth = st.sidebar.slider("要約を作成する階層の深さ", min_value=1, max_value=5, value=2)
                    content_depth = st.sidebar.slider("コンテンツを抽出する階層の深さ", min_value=1, max_value=5, value=2)

                    if st.button("文書構造を分析"):
                        with st.spinner("文書構造を分析中..."):
                            result = extract_and_summarize_level2_sections_multipass(
                                pages_images,
                                chunk_size=chunk_size,
                                summary_depth=summary_depth,
                                content_depth=content_depth,
                                file_name=file_name
                            )
                            st.json(result)

                with tab2:
                    st.header("画像分析")
                    st.write("PDFの各ページから表や図、グラフを抽出します。")
                    
                    # 分析ボタン
                    if st.button("すべてのページを分析"):
                        total_pages = len(pages_images)
                        progress_bar = st.progress(0)
                        
                        for i, page_info in enumerate(pages_images):
                            with st.spinner(f"ページ {i + 1}/{total_pages} を分析中..."):
                                # 連結成分ベースの分析を実行
                                analysis_result = analyze_image_content_with_connected_components(
                                    page_info["path"],
                                    file_name,
                                    i + 1
                                )
                                
                                # 結果を表示
                                with st.expander(f"ページ {i + 1} の分析結果"):
                                    display_single_analysis_results(
                                        analysis_result,
                                        st.session_state.page_manager,
                                        i + 1,
                                        file_name
                                    )
                                
                                # プログレスバーを更新
                                progress_bar.progress((i + 1) / total_pages)
                        
                        st.success("すべてのページの分析が完了しました")
                        
                        # 最終的なJSONファイルの内容を表示
                        json_filename = os.path.join(TMP_DIR, "analysis_results", file_name, "analysis_results.json")
                        if os.path.exists(json_filename):
                            with open(json_filename, "r", encoding="utf-8") as f:
                                final_results = json.load(f)
                            st.write("### 全ページの分析結果")
                            st.json(final_results)
                    
                    # 個別ページの分析オプションも残す
                    st.divider()
                    st.subheader("個別ページの分析")
                    
                    # ページ選択
                    page_num = st.selectbox(
                        "分析するページを選択",
                        options=list(range(1, len(pages_images) + 1)),
                        format_func=lambda x: f"ページ {x}"
                    )

                    # 個別分析ボタン
                    if st.button("選択したページを分析"):
                        with st.spinner("画像を分析中..."):
                            selected_image = pages_images[page_num - 1]
                            
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
                
                with tab3:
                    st.header("分析結果サマリー")
                    summary = st.session_state.page_manager.get_regions_summary()
                    if summary:
                        for page_num, page_summary in summary.items():
                            with st.expander(f"ページ {page_num}"):
                                st.write(f"総領域数: {page_summary['total_regions']}")
                                st.write(f"統合された領域数: {page_summary['merged_regions']}")
                                st.write("### 検出された領域")
                                for region in page_summary['regions']:
                                    region_type = region.get('type', '不明')
                                    st.write(f"- 領域 {region['id']} ({region_type})")
                                    st.write(f"  - バウンディングボックス: x={region['bbox']['x']}, y={region['bbox']['y']}, "
                                           f"width={region['bbox']['width']}, height={region['bbox']['height']}")
                                    if region['merged_from']:
                                        st.write(f"  - 統合元: 領域 {', '.join(map(str, region['merged_from']))}")
                    else:
                        st.info("まだ分析が実行されていません。")

            finally:
                # 処理完了のログ記録
                logger.info("PDFの処理が完了しました")
                # すべての処理が完了したら一時フォルダを削除
                for page_info in pages_images:
                    tmp_dir = page_info.get("tmp_dir")
                    if tmp_dir and os.path.exists(tmp_dir):
                        try:
                            shutil.rmtree(tmp_dir)
                            logger.info(f"一時フォルダを削除しました: {tmp_dir}")
                        except Exception as e:
                            logger.error(f"一時フォルダの削除に失敗しました: {str(e)}")

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.error(f"エラーが発生しました: {str(e)}")
            # エラー時の自動クリーンアップを削除

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
3. "should_merge_with": 1つの図/グラフ/表が複数の領域に分割されてしまっている場合、統合が必要なのでどのregionと統合すべきかを返してください.
4. "needs_split": 図/グラフ/表/テキストなど複数の要素が1つの領域（1つの画像）に含まれている場合、さらに分割が必要なのでtrueを返してください.
   - 1つの領域に複数の段落のテキスト、図表が含まれている場合
   - 複数のグラフが1つの領域として検出されている場合
   など

以下のJSON形式で回答してください：
{{
    "regions": [
        {{
            "id": 数値のID,  // "region1"ではなく1を返してください。"region1-1"のように枝番がある場合は1-1と返してください。
            "type": "図/グラフ/表/イメージ/テキスト/その他",
            "description": "内容の説明",
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
3. 分割が必要/不要な場合は、その理由を具体的に説明してください
4. 統合や分割の判断は、領域の内容や配置を考慮して行ってください
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
                if region["type"] not in ["図", "グラフ", "表", "その他"]:
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

def analyze_image_content_with_connected_components(image_path, file_name, page_num, is_subsplit=False):
    """
    連結成分解析を使用して画像から大きな領域を抽出する関数
    重複や近接している領域は統合します。
    Args:
        image_path: 分析対象の画像パス
        file_name: 分析対象のファイル名
        page_num: ページ番号
        is_subsplit: 再分割処理かどうか（Trueの場合はより細かい分割を行う）
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

        # 再分割時は異なるパラメータを使用
        if is_subsplit:
            min_area = (img_width * img_height) * 0.001  # 0.1%に下げる
            max_area = (img_width * img_height) * 0.5    # 50%に下げる
            blur_size = 3  # ブラーを弱める
            kernel_size = min(img_width, img_height) // 50  # より小さいカーネルを使用
            dilation_iterations = 1  # 膨張処理を弱める
        else:
            min_area = (img_width * img_height) * 0.005  # 画像全体の0.5%
            max_area = (img_width * img_height) * 0.95   # 画像全体の95%
            blur_size = 5
            kernel_size = min(img_width, img_height) // 30
            dilation_iterations = 3

        # カーネルサイズの制限
        kernel_size = max(5, min(kernel_size, 100))  # 最小5、最大100に制限

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
            if has_overlap and overlap_ratio > 0.001:  # 0.1%以上の重なり
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
            
            # 再分割時は距離の閾値を小さくする
            distance_threshold = min(diag1, diag2) * (0.3 if is_subsplit else 0.5)
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

        logger.info(f"領域 {region_info['id']} の分割処理を開始")

        # 領域を切り出し
        x, y = region_info["position"]["x"], region_info["position"]["y"]
        w, h = region_info["position"]["width"], region_info["position"]["height"]
        
        # 余白を追加
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(original_image.shape[1], x + w + margin)
        y2 = min(original_image.shape[0], y + h + margin)
        
        roi = original_image[y1:y2, x1:x2].copy()
        
        # 切り出した画像を一時ファイルとして保存
        temp_roi_path = os.path.join(TMP_DIR, f"temp_roi_{region_info['id']}.png")
        cv2.imwrite(temp_roi_path, roi)
        
        # 一時ファイルをBytesIOに変換（既存の処理が期待する形式に合わせる）
        with open(temp_roi_path, 'rb') as f:
            roi_bytes = io.BytesIO(f.read())
        
        # 既存の分析処理を実行（is_subsplit=Trueを指定）
        sub_regions_result = analyze_image_content_with_connected_components(
            temp_roi_path, 
            f"{file_name}_sub_{region_info['id']}", 
            f"{page_num}_{region_info['id']}",
            is_subsplit=True  # 再分割時は細かい分割を行う
        )
        
        if "error" in sub_regions_result:
            logger.error(f"サブ領域の分析でエラーが発生: {sub_regions_result['error']}")
            return [region_info]

        # サブ領域の座標は相対座標のまま保持
        logger.info(f"検出されたサブ領域の数: {len(sub_regions_result['figures'])}")
        
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

        if "error" not in sub_llm_analysis:
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
                return merged_result["figures"]

        # エラーが発生した場合やマージ提案がない場合は、元の分割結果を返す
        logger.info("マージ処理をスキップし、元の分割結果を返します")
        
        # 最後に親領域の座標を加算して返す
        for region in sub_regions_result["figures"]:
            region.update({
                "parent_id": region_info["id"],
                "split_depth": recursion_depth + 1,
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
        return [region_info]

def process_regions_with_split(analysis_result, llm_analysis, file_name, page_num, original_image):
    """
    統合と分割を含む領域処理のメイン関数
    """
    try:
        # 1. まず領域の統合を実行
        merged_result = merge_regions(analysis_result, llm_analysis, file_name, page_num)
        
        # st.json(merged_result)
        # st.json(llm_analysis)
        # 2. 分割が必要な領域を処理
        final_regions = []
        for region in merged_result["figures"]:
            region_analysis = next(
                (r for r in llm_analysis["regions"] if r["id"] == region["id"]),
                None
            )
            # st.write("region_analysis")
            # st.json(region_analysis)
            if region_analysis and region_analysis.get("needs_split", False):
                logger.info(f"領域 {region['id']} の分割を開始")
                # 領域を分割
                split_regions = split_region_using_existing_process(
                    region,
                    original_image,
                    file_name,
                    page_num
                )
                final_regions.extend(split_regions)
            else:
                final_regions.append(region)
        
        # st.write("final_regions")
        # st.json(final_regions)
        # 3. 結果を更新
        merged_result["figures"] = final_regions
        
        # 4. 結果の可視化
        visualize_final_regions(original_image, final_regions, file_name, page_num)
        
        return merged_result
        
    except Exception as e:
        logger.error(f"領域処理でエラーが発生: {str(e)}")
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