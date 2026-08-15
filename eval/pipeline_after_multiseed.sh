#!/bin/bash
# ═══════════════════════════════════════════════════════════
# 自动流水线：等待多seed完成 → B2鸿沟归因 → Phase C Sweep
# ═══════════════════════════════════════════════════════════
set -e

cd /home/luischen/文档/post-training-01
export HF_ENDPOINT=https://hf-mirror.com

echo "══════════════════════════════════════════════════════════"
echo "  流水线启动: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"

# ─── Step 1: 等待多 seed 实验完成 ───
echo ""
echo "[1/3] 等待多 seed 实验完成..."
while pgrep -f "run_multiseed.py" > /dev/null 2>&1; do
    sleep 60
done
echo "  ✅ 多 seed 实验已完成: $(date '+%H:%M:%S')"

# 短暂等待 GPU 释放
sleep 10

# ─── Step 2: Phase B2 鸿沟归因 ───
echo ""
echo "[2/3] 运行 Phase B2 鸿沟归因..."
python3 eval/gap_attribution.py 2>&1 | tee results/efficiency/B2_run.log
echo "  ✅ B2 完成: $(date '+%H:%M:%S')"

sleep 5

# ─── Step 3: Phase C 调优 Sweep ───
echo ""
echo "[3/3] 运行 Phase C 调优 Sweep..."
python3 eval/run_c_sweep.py 2>&1 | tee results/efficiency/C_sweep_run.log
echo "  ✅ Phase C 完成: $(date '+%H:%M:%S')"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  全部完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════════════════"
