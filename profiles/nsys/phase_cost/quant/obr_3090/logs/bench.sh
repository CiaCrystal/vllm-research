source /root/rivermind-data/env.sh; source $W/vllm/.venv/bin/activate; cd $W; mkdir -p prof res
export HF_HUB_OFFLINE=1
CASES="--case t1=1 --case t8=8 --case t64=64 --case t256=256 --case t1024=1024 --case t2048=2048 --case t8192=8192 --case d1=1*1@1024 --case d8=8*1@1024"
run(){ n=$1; m=$2; shift 2; echo "=== $n $(date +%T)"; vllm bench forward --model $m --load-format auto --max-model-len 8192 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --kv-len-jitter 0 --iters 40 --samples 5 $CASES --profile-dir $W/prof/$n --output-json $W/res/$n.json "$@" > $W/logs/bench_$n.log 2>&1; echo "rc=$? $(date +%T)"; }
run sparse $W/ckpt/int8_24_nocomp --linear-backend cutlass_sparse
run int8_24 $W/ckpt/int8_24_nocomp
run int8 $W/ckpt/int8_dense
run bf16 Qwen/Qwen3-8B
echo BENCH_DONE
