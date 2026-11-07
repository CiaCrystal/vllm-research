source /root/rivermind-data/env.sh; source $W/vllm/.venv/bin/activate; cd $W
until grep -q "^RC=" logs/obr_full.log; do sleep 10; done
grep -q "^RC=0" logs/obr_full.log || { echo OBR_FAILED; exit 1; }
for r in int8_obr bf16 int8 int4_obr int8_nocomp int8_obr_sparse; do
  echo "=== $r $(date +%T)"; python -I obr/scripts/eval_route.py $r; echo "rc=$? $(date +%T)"; sleep 3
done
echo EVAL_DONE
