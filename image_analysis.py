import cv2
import numpy as np
import streamlit as st

def extract_tables_from_image(image, threshold_value=128, horizontal_kernel_size=30, vertical_kernel_size=30, min_area=2500):
    """
    画像からテーブルを抽出する関数
    
    Args:
        image: 入力画像
        threshold_value: 二値化閾値
        horizontal_kernel_size: 水平カーネルサイズ
        vertical_kernel_size: 垂直カーネルサイズ
        min_area: 最小テーブル面積
    """
    # カラー画像をグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二値化
    _, bin_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # 処理過程を表示
    st.subheader("処理過程")
    col1, col2 = st.columns(2)
    with col1:
        st.text("グレースケール変換")
        st.image(gray, use_container_width=True)
    with col2:
        st.text("二値化処理")
        st.image(bin_img, use_container_width=True)
    
    # 水平線強調
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
    horizontal_lines = cv2.erode(bin_img, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=1)
    
    # 垂直線強調
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
    vertical_lines = cv2.erode(bin_img, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)
    
    # 水平線と垂直線を結合
    table_lines = cv2.add(horizontal_lines, vertical_lines)
    
    # 処理過程を表示 (続き)
    col1, col2 = st.columns(2)
    with col1:
        st.text("水平・垂直線検出")
        st.image(table_lines, use_container_width=True)
    
    # 輪郭検出
    contours, _ = cv2.findContours(table_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 結果表示用の画像
    result_img = image.copy()
    extracted_tables = []
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # 面積でフィルタリング
        if area > min_area:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            table_roi = image[y:y+h, x:x+w]
            extracted_tables.append((i, table_roi, (x, y, w, h)))
    
    with col2:
        st.text("検出結果")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # 抽出結果を表示
    if extracted_tables:
        st.subheader(f"抽出されたテーブル ({len(extracted_tables)}個)")
        cols = st.columns(min(3, len(extracted_tables)))
        
        for i, (idx, table_img, (x, y, w, h)) in enumerate(extracted_tables):
            with cols[i % 3]:
                st.image(cv2.cvtColor(table_img, cv2.COLOR_BGR2RGB), 
                         caption=f"テーブル {idx+1} (位置: x={x}, y={y}, 幅={w}, 高さ={h})",
                         use_container_width=True)
    else:
        st.info("テーブルが検出されませんでした。パラメータを調整してみてください。")

def extract_graphs_from_image(image, threshold_value=200, invert_binary=True, use_gaussian_blur=False, gaussian_kernel_size=5):
    """
    画像からグラフを抽出する関数
    
    Args:
        image: 入力画像
        threshold_value: 二値化閾値
        invert_binary: 二値化を反転するかどうか
        use_gaussian_blur: ガウスブラーを適用するかどうか
        gaussian_kernel_size: ガウスブラーのカーネルサイズ
    """
    # カラー画像をグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 必要に応じてガウスブラーを適用
    if use_gaussian_blur:
        gray = cv2.GaussianBlur(gray, (gaussian_kernel_size, gaussian_kernel_size), 0)
    
    # 二値化
    thresh_type = cv2.THRESH_BINARY_INV if invert_binary else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(gray, threshold_value, 255, thresh_type)
    
    # 処理過程を表示
    st.subheader("処理過程")
    col1, col2 = st.columns(2)
    with col1:
        st.text("グレースケール変換" + (" + ガウスブラー" if use_gaussian_blur else ""))
        st.image(gray, use_container_width=True)
    with col2:
        st.text("二値化処理")
        st.image(thresh, use_container_width=True)
    
    # 輪郭検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 結果表示用の画像
    result_img = image.copy()
    
    # 最大の輪郭を探す (グラフは通常大きな領域を占める)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 元画像に矩形を描画
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # グラフ領域の切り出し
        graph_img = image[y:y+h, x:x+w]
        
        # 処理結果を表示
        col1, col2 = st.columns(2)
        with col1:
            st.text("検出結果")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.text("抽出されたグラフ")
            st.image(cv2.cvtColor(graph_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        # グラフの詳細情報
        st.subheader("グラフ情報")
        st.write(f"位置: x={x}, y={y}")
        st.write(f"サイズ: 幅={w}, 高さ={h}")
        st.write(f"面積: {w*h} ピクセル")
    else:
        st.info("グラフが検出されませんでした。パラメータを調整してみてください。")

def extract_figures_from_image(image, threshold_value=128, morph_kernel_size=5, min_area=2500):
    """
    画像から図形を抽出する関数
    
    Args:
        image: 入力画像
        threshold_value: 二値化閾値
        morph_kernel_size: モルフォロジー演算のカーネルサイズ
        min_area: 最小図形面積
    """
    # カラー画像をグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二値化
    _, bin_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # 処理過程を表示
    st.subheader("処理過程")
    col1, col2 = st.columns(2)
    with col1:
        st.text("グレースケール変換")
        st.image(gray, use_container_width=True)
    with col2:
        st.text("二値化処理")
        st.image(bin_img, use_container_width=True)
    
    # モルフォロジー演算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    # 膨張
    dilated = cv2.dilate(bin_img, kernel, iterations=1)
    # 収縮
    morphed = cv2.erode(dilated, kernel, iterations=1)
    
    # 輪郭検出
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 結果表示用の画像
    result_img = image.copy()
    extracted_figures = []
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        
        # 面積でフィルタリング
        if area > min_area:
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            figure_roi = image[y:y+h, x:x+w]
            extracted_figures.append((i, figure_roi, (x, y, w, h)))
    
    # モルフォロジー処理と検出結果を表示
    col1, col2 = st.columns(2)
    with col1:
        st.text("モルフォロジー処理後")
        st.image(morphed, use_container_width=True)
    with col2:
        st.text("検出結果")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # 抽出結果を表示
    if extracted_figures:
        st.subheader(f"抽出された図形 ({len(extracted_figures)}個)")
        cols = st.columns(min(3, len(extracted_figures)))
        
        for i, (idx, figure_img, (x, y, w, h)) in enumerate(extracted_figures):
            with cols[i % 3]:
                st.image(cv2.cvtColor(figure_img, cv2.COLOR_BGR2RGB), 
                         caption=f"図形 {idx+1} (位置: x={x}, y={y}, 幅={w}, 高さ={h})",
                         use_container_width=True)
    else:
        st.info("図形が検出されませんでした。パラメータを調整してみてください。")

# 画像分析機能を実行する関数
def analyze_image(image_path, analysis_type, params=None):
    """
    画像分析を実行する関数
    
    Args:
        image_path: 画像のパス
        analysis_type: 分析タイプ（"テーブル抽出"、"グラフ抽出"、"図形抽出"）
        params: 分析パラメータの辞書
    """
    if params is None:
        params = {}
    
    # 画像読み込み
    image_np = cv2.imread(image_path)
    if image_np is None:
        st.error(f"画像の読み込みに失敗しました: {image_path}")
        return
    
    # 元画像を表示
    st.subheader("元の画像")
    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    # 選択された分析タイプに応じた処理
    if analysis_type == "テーブル抽出":
        extract_tables_from_image(
            image_np, 
            params.get("threshold_value", 128), 
            params.get("horizontal_kernel_size", 30), 
            params.get("vertical_kernel_size", 30), 
            params.get("min_table_area", 2500)
        )
    elif analysis_type == "グラフ抽出":
        extract_graphs_from_image(
            image_np, 
            params.get("threshold_value", 200), 
            params.get("invert_binary", True), 
            params.get("use_gaussian_blur", False), 
            params.get("gaussian_kernel_size", 5)
        )
    elif analysis_type == "図形抽出":
        extract_figures_from_image(
            image_np, 
            params.get("threshold_value", 128), 
            params.get("morph_kernel_size", 5), 
            params.get("min_figure_area", 2500)
        )
    else:
        st.error(f"未対応の分析タイプです: {analysis_type}")

# 画像分析タブのUI関数
def render_image_analysis_tab():
    """
    画像分析タブのUIをレンダリングする関数
    """
    st.header("画像分析")
    
    # 画像分析が利用可能なのはオンデマンド処理の場合のみ
    if 'pdf_doc' not in st.session_state:
        st.warning("PDFファイルをアップロードしてください。")
        return

    # ページ選択
    page_num = st.number_input("分析するページ番号", min_value=1, max_value=st.session_state.total_pages, value=1)
    
    # 分析方法の選択
    analysis_type = st.radio(
        "分析タイプを選択",
        ["テーブル抽出", "グラフ抽出", "図形抽出"],
        index=0
    )
    
    # サイドバーにパラメータ設定を追加
    with st.sidebar.expander("画像分析パラメータ", expanded=True):
        params = {}
        
        if analysis_type == "テーブル抽出":
            st.sidebar.subheader("テーブル抽出設定")
            params["threshold_value"] = st.sidebar.slider("二値化閾値", min_value=0, max_value=255, value=128, step=1)
            params["horizontal_kernel_size"] = st.sidebar.slider("水平カーネルサイズ", min_value=5, max_value=100, value=30, step=5)
            params["vertical_kernel_size"] = st.sidebar.slider("垂直カーネルサイズ", min_value=5, max_value=100, value=30, step=5)
            params["min_table_area"] = st.sidebar.slider("最小テーブル面積", min_value=500, max_value=10000, value=2500, step=500)
        
        elif analysis_type == "グラフ抽出":
            st.sidebar.subheader("グラフ抽出設定")
            params["threshold_value"] = st.sidebar.slider("二値化閾値", min_value=0, max_value=255, value=200, step=1)
            params["invert_binary"] = st.sidebar.checkbox("二値化を反転", value=True)
            params["use_gaussian_blur"] = st.sidebar.checkbox("ガウスブラーを適用", value=False)
            params["gaussian_kernel_size"] = st.sidebar.slider("ガウスブラーカーネルサイズ", min_value=3, max_value=11, value=5, step=2) if params["use_gaussian_blur"] else 5
        
        elif analysis_type == "図形抽出":
            st.sidebar.subheader("図形抽出設定")
            params["threshold_value"] = st.sidebar.slider("二値化閾値", min_value=0, max_value=255, value=128, step=1)
            params["morph_kernel_size"] = st.sidebar.slider("モルフォロジーカーネルサイズ", min_value=1, max_value=10, value=5, step=1)
            params["min_figure_area"] = st.sidebar.slider("最小図形面積", min_value=500, max_value=10000, value=2500, step=500)
    
    # 実行ボタン
    if st.button("分析実行"):
        try:
            with st.spinner(f"ページ {page_num} を処理中..."):
                # DPI設定とメモリ設定を取得
                dpi_value = st.session_state.get("dpi_value", 300)
                max_pixels = st.session_state.get("max_pixels", 4000000)
                
                # ページ画像を取得
                page_image_info = get_page_image(
                    st.session_state.pdf_doc, 
                    page_num - 1, 
                    dpi=dpi_value, 
                    max_pixels=max_pixels,
                    session_tmp_dir=st.session_state.session_tmp_dir
                )
                
                # 画像分析を実行
                analyze_image(page_image_info['path'], analysis_type, params)
                
        except Exception as e:
            st.error(f"画像分析中にエラーが発生しました: {str(e)}")
            import logging
            logging.error(f"画像分析エラー: {str(e)}", exc_info=True)
    
    # 使用方法ガイド
    with st.expander("使用方法"):
        st.markdown("""
        ### 画像分析の使い方
        
        1. **分析するページ番号**を選択
        2. **分析タイプ**を選択
            - テーブル抽出: 表を検出して抽出します
            - グラフ抽出: グラフを検出して抽出します
            - 図形抽出: さまざまな図形を検出して抽出します
        3. サイドバーの**画像分析パラメータ**で設定を調整
        4. **分析実行**ボタンをクリック
        
        #### パラメータの説明
        - **二値化閾値**: 画像を白黒に変換する際の境界値。低いと暗い部分も白に、高いと明るい部分も黒になります。
        - **カーネルサイズ**: 形状検出に使用される構造要素のサイズ。大きいとより大きな特徴を検出します。
        - **最小面積**: この値より小さい検出結果を無視します。ノイズ除去に役立ちます。
        """)

# app.pyから呼び出す必要がある関数の定義
def get_page_image(doc, page_num, dpi=1000, max_pixels=4000000, session_tmp_dir=None):
    """
    この関数はapp.pyから参照されるためのスタブです。
    実際の実装はapp.pyに存在し、このファイルからはそのまま呼び出します。
    """
    # この関数はapp.pyからインポートして使用するため、
    # ここでの実装は必要ありません
    pass 