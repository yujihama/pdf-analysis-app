# PDF分析アプリ

このアプリケーションは、PDFファイルをアップロードして文書構造を分析し、図表や画像を抽出するStreamlitアプリケーションです。

## 機能

- PDFファイルのアップロードと処理
- 文書構造の分析（章や節の抽出）
- 図表や画像の自動検出と分類
- 検出された要素の要約生成

## 環境設定

1. リポジトリをクローンします
2. 必要なパッケージをインストールします
   ```
   pip install -r requirements.txt
   ```
3. `.env.sample`ファイルを`.env`にコピーし、APIキーなどの設定を行います

## 環境変数の設定

`.env`ファイルに以下の環境変数を設定します：

### OpenAI API（通常のOpenAI APIを使用する場合）

```
OPENAI_API_KEY=your_openai_api_key_here
```

### Azure OpenAI API（Azure OpenAI APIを使用する場合）

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name_here
AZURE_OPENAI_API_VERSION=2023-05-15
```

**注意**: Azure OpenAI APIを使用する場合は、上記の4つの環境変数をすべて設定する必要があります。設定されていない場合は、通常のOpenAI APIが使用されます。

### その他の設定

```
ENABLE_FILE_LOGGING=false  # ログファイルを出力する場合はtrueに設定
```

## 使用方法

1. アプリケーションを起動します
   ```
   streamlit run app.py
   ```
2. ブラウザで表示されるインターフェースからPDFファイルをアップロードします
3. 「文書構造を分析」または「すべてのページを分析」ボタンをクリックして分析を開始します
4. 分析結果が表示されます

## 注意事項

- 大きなPDFファイルの処理には時間がかかる場合があります
- 画像分析には十分なメモリが必要です
- APIキーの使用には料金が発生する場合があります 