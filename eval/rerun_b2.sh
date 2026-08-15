#!/bin/bash
# B2 鸿沟归因重跑（等待 Phase C 完成后执行）
set -e

cd /home/luischen/文档/post-training-01
export HF_ENDPOINT=https://hf-mirror.com

echo "══════════════════════════════════════════════════════════"
echo "  B2 鸿沟归因重跑 | 启动: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"

# 等待 Phase C 完成
echo "[等待] Phase C sweep 完成..."
while pgrep -f "run_c_sweep.py" > /dev/null 2>&1; do
    sleep 60
done
echo "  ✅ Phase C 已完成: $(date '+%H:%M:%S')"

sleep 10

# 运行 B2
echo ""
echo "[运行] B2 鸿沟归因 (batch=2, seq=256)..."
python3 eval/gap_attribution.py 2>&1 | tee results/efficiency/B2_rerun.log

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  B2 完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"
