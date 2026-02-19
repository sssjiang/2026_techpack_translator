# 配置说明

本目录包含 `config.yaml`（实际使用的配置）和 `config example.yaml`（配置示例与模板）。  
首次使用可复制 `config example.yaml` 为 `config.yaml`，再按需修改。

---

## 一、快速开始

1. **若还没有 config.yaml**
   ```bash
   cp "config example.yaml" config.yaml
   ```

2. **必须配置的项**
   - **OCR**：若使用 `qwen_ocr`，在 `ocr.api_key` 中填写阿里云百炼 API Key（或设置环境变量 `DASHSCOPE_API_KEY`）。
   - **翻译**：若使用 `deepl` 或 `google_cloud`，在 `translation.api_key` 中填写对应 API Key（或使用下方环境变量）。

3. **运行**
   ```bash
   python3 main.py input/图片.png output/输出.png
   ```

---

## 二、OCR 配置（ocr）

| 项 | 说明 |
|----|------|
| **engine** | `qwen_ocr`（阿里云，带定位）/ `paddleocr` / `tesseract` / `easyocr` |
| **api_key** | 仅 **qwen_ocr** 需要。阿里云百炼 API Key，或环境变量 `DASHSCOPE_API_KEY` |
| **base_url** | 仅 qwen_ocr。北京默认即可；弗吉尼亚/新加坡见 config example 内注释 |
| **model** | 一般保持 `qwen-vl-ocr-latest` |
| **enable_rotate** | 是否自动矫正倾斜图像 |

本地引擎（paddleocr / tesseract / easyocr）无需 API Key，安装对应依赖即可。

---

## 三、翻译配置（translation）

| 项 | 说明 |
|----|------|
| **engine** | `google`（免费）/ `google_cloud` / `deepl` / `local` |
| **source_lang** | 源语言，如 `en`（英文） |
| **target_lang** | 目标语言，如 `zh`、`zh-cn`（中文） |
| **api_key** | **google** 可不填；**google_cloud** / **deepl** 需填写，或使用环境变量 |

**API Key 环境变量（可选，避免写进配置文件）：**

- DeepL：`DEEPL_API_KEY`
- Google Cloud 翻译：`GOOGLE_CLOUD_API_KEY`

**DeepL**：免费版 Key 以 `:fx` 结尾，使用 `https://api-free.deepl.com`。

---

## 四、其他常用项

- **detection.keywords**：用于识别“设计包图像”区域的文案，可按需增删。
- **rendering.fonts**：渲染翻译文字使用的字体；若中文乱码，请在项目 `fonts/` 下放入中文字体（见项目根目录 `fonts/README.md`）。
- **output.generate_preview**：是否生成原图与译图对比图。
- **logging.level**：日志级别，如 `DEBUG`、`INFO`。

---

## 五、配置文件位置

- 默认读取：项目根目录下的 `config/config.yaml`。
- 自定义路径：运行时的 `--config` 参数，例如：
  ```bash
  python3 main.py input.png output.png --config /path/to/my_config.yaml
  ```
