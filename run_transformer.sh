#!/bin/bash
# 运行 Transformer 召回模型（后台运行）
nohup python -u news3_transformer.py > news3_transformer.log 2>&1 &
echo "进程已启动，PID: $!"
echo "日志文件: news3_transformer.log"
echo "查看日志: tail -f news3_transformer.log"



nohup python -u news5.py > news5.log 2>&1 &