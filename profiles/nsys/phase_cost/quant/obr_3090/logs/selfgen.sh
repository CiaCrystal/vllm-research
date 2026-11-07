# Wait until the profiling run has left the GPU, then generate, calibrate and evaluate.
source /root/rivermind-data/env.sh; source $W/vllm/.venv/bin/activate; cd $W
idle=0
while [ $idle -lt 9 ]; do
  busy=$(pgrep -f "[p]rof_serve|bench [f]orward|bin/[v]llm serve" | wc -l)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  recent=$(find $W/roofline -newermt "-5 minutes" 2>/dev/null | wc -l)
  if [ "$busy" -eq 0 ] && [ "$mem" -lt 500 ] && [ "$recent" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  sleep 20
done
echo "GPU_IDLE $(date +%T)"
cd $W/obr/scripts
HF_HUB_OFFLINE=1 python -I gen_calib.py > $W/logs/gen_calib.log 2>&1; echo "GEN_RC=$? $(date +%T)"
grep -E "^CALIB" $W/logs/gen_calib.log
HF_HUB_OFFLINE=1 python -I obr_8b.py --maxit 60 --calib $W/data/calib_selfgen.pt --root $W/ckpt --tag _selfgen > $W/logs/obr_selfgen.log 2>&1; echo "OBR_RC=$? $(date +%T)"
cd $W
for r in int8_obr_selfgen int4_obr_selfgen; do
  echo "=== $r $(date +%T)"; python -I obr/scripts/eval_route.py $r --skip-stream --port 8022 2>&1 | grep -v "^\[transformers"; echo "rc=$? $(date +%T)"; sleep 3
done
echo SELFGEN_DONE
