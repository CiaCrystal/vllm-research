# 任务：用现有资源分析 Qwen3-8B 在 RTX 3090 上的性能瓶颈、开销和 roofline

你在本地 WSL 的 `/home/bca/code/vllm`（分支 `research/nvtx-profiling`）工作，通过 `ssh gpu3090` 操作一台按量计费的 RTX 3090 服务器。请控制时长，每完成一步汇报结果和已用时间。

## 先读

- `AGENTS.md`（其中 WSL 路径只适用于本机）
- `.agents/research/nvtx-profiling-changelog.md`：「不要重新踩的坑」，尤其第 15、17、18、29、35 条和第 47 至 62 条，以及 CHG-012 至 CHG-016
- `profiles/nsys/phase_cost/quant/obr_3090/report.md`：已有的速度和精度数据
- `profiles/nsys/phase_cost/quant/obr_3090/forward/`：`vllm bench forward` 的结果和每个 kernel 的 GPU 时间（`prof/<路线>/<case>.rank0.kernels.json`），可以直接复用

## 服务器环境（已就绪，不要重装）

- 环境变量：`source /root/rivermind-data/env.sh`（设置 `W=/root/rivermind-data`、uv、HF 缓存、镜像）
- vLLM：`$W/vllm`，venv 是 `$W/vllm/.venv`。代码是基线 `7dca18e206` 加本分支的补丁，不是 Git checkout。本地改动要用小补丁同步，上行只有约 3KB/s。
- 模型：`Qwen/Qwen3-8B`，要设 `HF_HUB_OFFLINE=1`
- checkpoint：`$W/ckpt/int8_dense`（稠密 INT8）、`/root/obr_ckpt/int8_24_obr`（INT8 + 2:4 + OBR）
- 稀疏 kernel：`--linear-backend cutlass_sparse`，已编译在 `$W/vllm/vllm/_sparse_int8_sm80.so`
- 现成脚本：`$W/logs/serve.sh`、`$W/logs/wait_serve.sh`、`$W/obr/scripts/eval_route.py`（流式 A、D、F 测时）
- 驱动 580.119.02，CUDA 13.0，torch 2.14.0+cu130，20 vCPU，100GiB 内存
- 没有 tmux。长任务用 `nohup setsid`，输出写文件。

## 约束

- **nsys 不可用**：镜像删掉了程序文件，只剩 dpkg 记录。**不要安装或重装任何东西。** 只能用 torch profiler（`--profiler-config`）、`vllm bench forward --profile-dir`、Proton（如果可用）和 PyTorch 自身。
- 每组合只测 3 次，1 轮。
- 计时时 GPU 上不能有其他任务。开始前确认 `nvidia-smi` 显存为 0。
- 不 commit、不 push。原始 trace（`.json.gz`）不进 Git，只把汇总表和小的 JSON 放进 `profiles/nsys/phase_cost/quant/obr_3090/profiling/`。
- 每次改代码都在 changelog 末尾追加 CHG，新发现的坑也写进去。
- 远程命令不要用会匹配到自己命令行的 `pkill -f`（坑 53）。
- 只报告数据支持的结论。推测要标注，没测的要写明未测。

## 路线和负载

- 路线：bf16、稠密 INT8（`int8_dense`）、稀疏 INT8（`int8_24_obr` + `cutlass_sparse`）
- 服务参数：`--gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-model-len 2048 --max-num-batched-tokens 8192`
- 负载：A（256 token × 1，生成 33），D（1024 × 8，生成 33），F（256 × 8，生成 129），流式，每组合 3 次

## 步骤

1. **测实际可达的硬件上限（roof）。** 用 PyTorch 测：
   - 大形状 bf16 matmul 的 TFLOPS
   - INT8 GEMM 的 TOPS（`vllm._custom_ops.cutlass_scaled_mm`）
   - 设备内拷贝带宽（GB/s）
   
   和官方规格对比。以下规格凭记忆，需核对 NVIDIA GA102 白皮书：显存带宽 936 GB/s，FP16/BF16 Tensor（FP32 累加）71 TFLOPS，INT8 Tensor 284 TOPS（稀疏 ×2）。

2. **kernel 级 roofline。** 优先复用 `forward/` 里已有的 kernels.json（bf16、int8、int8_24、sparse、sparse_col；case 为 1、8、64、256、1024、2048、8192 token 和 decode d1、d8）。不够再补跑 `vllm bench forward --iters 3 --samples 1 --kv-len-jitter 0 --profile-dir ...`。
   
   按 kernel 类别归类：线性层（区分 qkv、o、gate_up、down 和 lm_head）、注意力、激活量化、norm、激活函数、rotary、KV 写入、其他。按形状解析计算每类的 FLOPs 和字节数（权重、激活、KV），得到算术强度、实际 TFLOPS 和 GB/s，以及占 roof 的百分比。kernel 名称以 `example` 字段核对，不要只看 `name`（坑 15）。

3. **serving 级开销。** 用 `vllm serve ... --profiler-config '{"profiler":"torch","torch_profiler_dir":"<绝对路径>","torch_profiler_with_stack":false}'`，在 A、D、F 前后调 `/start_profile`、`/stop_profile`。每段只采 3 次请求，必须调 `/stop_profile`。从 trace 里拆出：
   - 每个 step 的 CPU 时间和 GPU 忙碌时间
   - GPU 空闲间隙
   - 调度、采样（`sample_tokens`）、输入准备、HTTP 和 detokenize 的占比
   - Decode 是否走了 CUDA Graph

   先确认 torch profiler 能否看到 CUDA Graph 里的 kernel。看不到就写明，Decode 的 kernel 细分改用步骤 2 的 bench 数据。

4. **把时间对上。** 解释每个配置的 TTFT 和 TPOT 由哪些部分组成：GPU 前向、CPU 开销、空闲、采样。各部分之和要和端到端测量对得上，对不上要写明差多少、可能为什么。

5. **结论与图。** 输出 roofline 图（bf16、INT8、稀疏的 Prefill 和 Decode 各一组点）。如果画图需要安装库，就只给 CSV 和表格，不安装。

## 需要明确回答的问题

1. Prefill 和 Decode 各自是计算受限、带宽受限，还是 CPU/调度受限？依据是什么？
2. Decode 每步的耗时，离“读一遍权重和 KV 的带宽下限”还差多少？差距在哪里？
3. 线性层以外，注意力、量化、norm 等占多少？下一步最值得优化的是哪一项？
4. 稀疏 kernel 在 roofline 上的位置：它省掉的是计算还是带宽？为什么大批量几乎没有收益（参考坑 50、56）？
5. CPU 侧开销（调度、采样、Python、HTTP）在 A、D、F 中各占多少？

## 已知数据，供核对，不要直接当结论

- 端到端（每组合 3 次）TPOT：bf16 20～22 ms，INT8 12.5～14.3 ms，稀疏 10.7～13.0 ms
- 8192 token 单次前向：bf16 1960 ms，其中线性 GEMM 1553 ms、Flash Attention 336 ms；INT8 1014 ms，GEMM 534 ms
- `Model loading took`：bf16 15.27 GiB，INT8 8.8 GiB，稀疏 9.65 GiB

## 交付（用中文）

1. `profiles/nsys/phase_cost/quant/obr_3090/profiling/report.md`：硬件上限、kernel 分类表、roofline 表或图、serving 开销拆解、上面 5 个问题的回答、实际命令、未做的部分。
2. changelog 追加 CHG，新坑加入「不要重新踩的坑」。
3. 已用时间，以及服务器上新增的文件和哪些可以删。
