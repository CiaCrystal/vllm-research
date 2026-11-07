"""Run tests/evals/gsm8k/gsm8k_eval.py unchanged, also dumping per-question answers."""
import importlib.util, json, sys
EVAL = "/root/rivermind-data/vllm/tests/evals/gsm8k/gsm8k_eval.py"
dump = sys.argv[1]
sys.argv = [EVAL] + sys.argv[2:]
spec = importlib.util.spec_from_file_location("gsm8k_eval", EVAL)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
orig = m._score_gsm8k
def score(states, output_tokens, labels, *a, **k):
    preds = [m.get_answer_value(s) for s in states]
    json.dump(dict(states=states, labels=[int(x) for x in labels], preds=[int(x) for x in preds],
                   output_tokens=output_tokens), open(dump, "w"))
    return orig(states, output_tokens, labels, *a, **k)
m._score_gsm8k = score
m.main()
