FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV 及中文字体）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    fonts-noto-cjk \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements 并安装 Python 依赖（仅 Qwen-OCR + DeepL）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 300

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs fonts

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 默认命令
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
