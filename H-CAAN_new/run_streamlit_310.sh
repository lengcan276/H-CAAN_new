#!/bin/bash
# 使用当前的conda环境Python
PYTHON_PATH=$(which python)
echo "Using Python: $PYTHON_PATH"

# 将正确的模块路径添加到PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# 运行streamlit应用
$PYTHON_PATH -m streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 8502 --server.headless true
