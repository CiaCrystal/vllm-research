source /root/rivermind-data/env.sh; source $W/vllm/.venv/bin/activate; cd $W/vllm
tools/build_sparse_int8_sm80.sh 2>&1 | grep -E "error" | head -3
cd $W/obr; python -I sparse_unit.py 1 3 8 16 17 64 1024 2>&1 | grep -v "^W1107" | tail -4
export HF_HUB_OFFLINE=1; cd $W
CASES="--case t1=1 --case t8=8 --case t64=64 --case t256=256 --case t1024=1024 --case t2048=2048 --case t8192=8192 --case d1=1*1@1024 --case d8=8*1@1024"
vllm bench forward --model $W/ckpt/int8_24_nocomp --load-format auto --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --kv-len-jitter 0 --iters 3 --samples 1 $CASES --profile-dir $W/prof/sparse_col --output-json $W/res/sparse_col.json --linear-backend cutlass_sparse > $W/logs/bench_sparse_col.log 2>&1
echo BENCH2 rc=$?; date +%T
