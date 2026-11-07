source /root/rivermind-data/env.sh; source $W/vllm/.venv/bin/activate; cd $W/obr/scripts
HF_HUB_OFFLINE=1 python -I mix_calib.py 2>&1 | grep -v "^\[transformers"; echo "MIXGEN_RC=$? $(date +%T)"
HF_HUB_OFFLINE=1 python -I obr_8b.py --maxit 60 --calib $W/data/calib_mix.pt --root $W/ckpt --root4 /root/obr_ckpt --tag _mix > $W/logs/obr_mix.log 2>&1; echo "OBR_RC=$? $(date +%T)"
cd $W
for r in int8_obr_mix int4_obr_mix; do
  echo "=== $r $(date +%T)"; python -I obr/scripts/eval_route.py $r --skip-stream --port 8022 2>&1 | grep -v "^\[transformers"; echo "rc=$? $(date +%T)"; sleep 3
done
echo MIX_DONE
