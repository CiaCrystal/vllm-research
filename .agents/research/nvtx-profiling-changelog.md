# NVTX 分层插桩：改动记录与交接

> 读者：接手 `research/nvtx-profiling` 分支的 agent。
> 用法：动手前读完「不要重新踩的坑」；**每次改动后在「改动记录」末尾追加一条**（模板在该节开头）。
> 旧条目不改写。发现旧结论有误时，追加新条目并写明更正了哪一条。

## 当前状态

| 项目 | 内容 |
| ---- | ---- |
| 分支 | `research/nvtx-profiling` |
| 基线提交 | `d038985573`（Ignore local profiling artifacts） |
| 提交状态 | CHG-011 及之前由用户提交并 push（`e26b649a68`）。CHG-012 至 CHG-021 由用户要求提交并 push |
| 审阅状态 | 用户已于 2026-11-06 接受 CHG-001 至 CHG-004 |
| 方向（2026-11-07） | OBR 2:4 稀疏实验在 CHG-020 结束，后续工作围绕 bf16 展开。3090 服务器上的量化 checkpoint 已删除，见 CHG-021 |
| 功能 | `VLLM_NVTX_ENABLE=1` 时输出 `Step/{Prefill,Decode,Mixed}` 和 `{阶段}/Layer_NN/{Attention,FFN}` |
| 已验证模型 | Qwen3-0.6B（28 层），`Qwen/Qwen3-0.6B`，已在本地 HF 缓存 |
| 最近一次测量 | 2026-11-06，eager 见 CHG-006，非 eager 基线和量化可行性见 CHG-007，量化对比见 CHG-008，TTFT 和 TPOT 见 CHG-009，OBR 可行性见 CHG-010，OBR 精度和稀疏 kernel 见 CHG-011。报告在 `profiles/nsys/phase_cost/report.md`、`profiles/nsys/phase_cost/noneager/report.md`、`profiles/nsys/phase_cost/quant/report.md` 、`profiles/nsys/phase_cost/quant/sparsity/report.md` 和 `profiles/nsys/phase_cost/quant/obr_3090/report.md` |

本任务改动的文件：

- `vllm/utils/nvtx_phase_ranges.py`（新增）
- `vllm/envs.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `docs/contributing/profiling.md`
- `.agents/research/nvtx-profiling-changelog.md`（本文件）

OBR 稀疏 kernel 实验新增的文件（CHG-011，未提交）：

- `csrc/quantization/w8a8/cutlass/c3x/sparse_fp8_mm_sm120.cu`（新增）
- `tools/build_sparse_fp8_sm120.sh`（新增）
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass_sparse.py`（新增）
- `vllm/model_executor/kernels/linear/__init__.py`
- `vllm/config/kernel.py`

工作区里**不属于本任务**、必须保留的内容：`AGENTS.md` 的本地修改（用户写的本地环境约定）和 `profiles/`（用户的 profiling 产物）。不要 reset、checkout 或 stash。

## 必须遵守的约束

来自用户的任务要求和 `AGENTS.md`：

- 不安装、升级或卸载任何包；不改 `.venv`、CUDA 和驱动。
- Python 只用 `/home/bca/code/vllm/.venv/bin/python`。
- 不 commit、不 push；原始 profiling 文件不进 Git。
- 只做 Profiling，不做量化、剪枝、算子融合等优化，不改模型计算逻辑。
  - 例外（用户 2026-11-06 要求）：允许做量化对比测量，方式限于 CLI 参数和放在仓库外的量化 checkpoint。仍不改源码。
  - 例外（用户 2026-11-07 批准）：允许复现 OBR 精度，并新增 2:4 稀疏 kernel（含 C++）。这只针对这项实验，不代表其他优化也放开。
- 热路径禁止 GPU 同步或拷贝：`synchronize`、`.item()`、`.cpu()`、`.tolist()`。
- WSL 下运行需要：

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nv_dispi.inf_amd64_20f49c6eb5c9b39b:$LD_LIBRARY_PATH
```

离线使用模型时再加 `HF_HUB_OFFLINE=1`。

`ruff`、`pre-commit`、`pytest` 都不在 `.venv` 里，且不能安装。截至目前 **lint 和仓库单测都没有跑过**。

## 不要重新踩的坑

这些都已实测确认，无需重新推导：

1. **实际路径是 Model Runner V2。** Qwen3 走 `vllm/v1/worker/gpu/model_runner.py`，不是 `gpu_model_runner.py`。实测 `use_v2_model_runner=True`、`async_scheduling=True`，eager 和默认模式都是如此。V1 runner 只是回退路径（`VLLM_USE_V2_MODEL_RUNNER=0` 或 V2 不支持的特性）。
2. **warmup 不走 `dummy_run`。** `warmup_kernels` 用合成的 `SchedulerOutput` 调真实的 `execute_model`（`dummy_run=False`），`batch_req_state` 非空。这条路径在 `--enforce-eager` 下也会执行。只判断 `dummy_run` 排除不了它，每个 trace 会固定多出 6 个 Step（Prefill 2、Decode 3、Mixed 1）。
3. **模型代码里不能写无条件的 NVTX。** vLLM 用 `fullgraph=True` 编译，Dynamo 遇到 `nvtx.range_push` 直接抛 `Unsupported`（实测 torch 2.14）。`fullgraph=False` 时才只是 graph break。
4. **`--enable-layerwise-nvtx-tracing` 在 V2 上不生效。** V2 runner 里没有接入 `PytHooks`，实测 0 条记录。不能用它和本功能做共存测试。
5. **`temperature=0` 跨进程不是逐位确定的。** 改动前的代码 5 次运行中有 2 次输出不同。比较输出时要有多次运行作对照，不能把单次差异直接归因于代码改动。
6. **吞吐噪声约 ±5%。** 同一批次内波动 3.5～5.7%，不同批次间曾出现 -6.76%。比较时要交替运行，不要跨批次比。
7. **`--enforce-eager` 关闭 JIT warmup。** 首次调用可能很慢：`Decode/Layer_00/Attention` 平均 3.4 ms，中位数 0.36 ms，最大 110 ms。分析时看中位数。冷启动是推断，未单独验证。
8. **`Step/*` 不包含 logits 和采样。** 它们在 `sample_tokens` 里。`input_layernorm`、`post_attention_layernorm` 也不在 layer range 内。
9. **关闭插桩的 trace 里仍有 23 条 NVTX 记录。** 这些是 CUB 自带的 `cub::DeviceScan::*`，不是本功能的标签。
10. **`--gpu-memory-utilization 0.5` 下必须限制 `--max-model-len`。** 默认 40960 需要 4.38 GiB KV，RTX 5070 只留出 4.04 GiB，引擎启动失败，错误在 `_check_enough_kv_cache_memory`。实测用 `--max-model-len 2048`。
11. **`vllm serve` 在这张卡上 `max_num_batched_tokens` 默认只有 2048，且默认开启 Chunked Prefill。** 8×1024 token 会被切块，序列是 `PMMMMD…`。需要 `--max-num-batched-tokens 8192`。
12. **一个请求的 `prompt` 传多条，并不会同时到达引擎。** 请求逐条发送，第一条先被单独调度，序列是 `PMD…`。可靠的对齐方式是不改代码的门控：`POST /pause?mode=keep&clear_cache=false`，发请求，等 0.3 s，`POST /resume`。这要设 `VLLM_SERVER_DEV_MODE=1`，同时加 `--host 127.0.0.1`，不要把开发接口暴露出去。
13. **异步调度下，Prefill 的 GPU 积压不在 Prefill 之后等待。** CPU 先发射第一个 Decode，等待落在它之后，且不属于任何 Step：D（8×1024）为 128.6 ms，C（8×256）为 10.0 ms，单请求配置没有。前两个 Decode 的 GPU 投影也会被压缩（D 为 5.3、8.9 ms），要看中位数。
14. **eager 下 CPU range 占比不反映计算量。** token 数增加 32 倍，Step CPU 仍是 11～12 ms，Attention、FFN 占比基本不变。判断 GPU 开销要看 kernel 时间。投影时间包含 kernel 之间的空闲（Decode 投影 11 ms，kernel 只有 2～4 ms），用 Step 投影减各层投影得到的“层外”也包含层间空闲，不能当作层外开销。
15. **`nsys stats :base` 的 kernel 名称不足以判断身份。** `Kernel2` 是 CUTLASS GEMM，`kernel` 是 cuBLAS gemvx，都是线性层的矩阵乘；同一层在批大小 1 和 8 的 Decode 里走的路径不同。要用 sqlite 的 `demangledName` 确认。
16. **`execute_context_N(tokens)_generation_M(n)` 标签可以独立核对批内容。** 这是 vLLM 自带的 NVTX 记录，能确认每个 Prefill step 的请求数和 token 数，不依赖本功能的标签。其中 `execute_context_0(0)_generation_0(0)` 是空 step，不产生 `Step/*`。
17. **CUDA Graph 默认按整图追踪，看不到里面的 kernel。** 要看 kernel 用 `--cuda-graph-trace=node`。此时同一个 `correlationId` 对应多个 kernel，按 `correlationId` 建字典会丢数据，要存成列表。本次 `node` 和 `graph` 模式的请求耗时差异在噪声内。
18. **非 eager 下 `Step/*` 的 CPU 时间不代表开销。** Decode 的 Step 只有约 0.06 ms，就是一次 `cudaGraphLaunch`。要看 GPU 时间线和完整周期。`max_cudagraph_capture_size` 是 512，所以超过 512 token 的 Prefill 不走 CUDA Graph，kernel 逐个发射。
19. **没有层标签时，kernel 可以按每层固定顺序归类。** 每层依次是 `qkv_proj`、rotary、KV 写入、注意力、`o_proj`、norm、`gate_up_proj`、激活、`down_proj`、norm。q/k norm 和 rotary 被融合，拆不开。
20. **请求窗口不能以下一轮 Step 的开始作为结束。** 下一轮 Step 之前的输入上传会被算进来，把 0.3 s 门控也带进窗口。曾因此把 A 的跨度算成 419 ms，实际是 97 ms。
21. **Qwen3 没有在线 INT8。** `_ONLINE_LINEAR_METHODS` 里没有 INT8，`int8_per_channel_weight_only` 只作用于 MoE，而 Qwen3 没有 MoE。用它不会量化任何线性层。这是读代码的结论，未实际运行。
22. **RTX 5070（capability 12.0，SM120）不支持 CUTLASS INT8。** 实测报错 `Int8 not supported on SM120`。`CutlassInt8ScaledMMLinearKernel.is_supported` 却返回 True，所以默认选中它，到运行时才失败。唯一的路径是 `--linear-backend triton`，且需要 INT8 checkpoint。
23. **CUTLASS W4A8 只支持 capability 9.0。** `is_device_capability(90)` 是精确匹配，这张卡返回 False，要判断“不低于”需用 `has_device_capability`。INT4 在这张卡上只剩 W4A16 系列，而且需要 checkpoint。
24. **直接调用 `cutlass_scaled_mm` 时，权重必须是列主序。** 用 `w.t()`，不要再加 `.contiguous()`，否则在 `scaled_mm_entry.cu:232` 报错。这个错和数据类型是否受支持无关，曾因此误判。
25. **`--quantization mxfp4` 在 SM120 上启动失败。** 报错 `SM12x cute-dsl requires NVFP4 with 128x4 scales`，发生在 `profile_run` 的首次前向，来自 FlashInfer。`cutlass_scaled_mm_supports_fp4(120)` 返回 True 和 `has_flashinfer_cutedsl()` 可用都不代表它能跑。
26. **MXFP8 的 FlashInfer autotune 在 nsys 下慢约 8 倍，且每次启动都重跑。** Qwen3 要 224 组：不带 nsys 约 5 分钟，带 nsys 约 22 分钟只完成 136 组。本机结果没有持久化，`~/.cache/flashinfer/autotune` 不存在。带 nsys 时用 `--disable-flashinfer-autotune`。不带 nsys 的 A、D、F 对照里，开关 autotune 的请求耗时差异在 3.2% 以内。
27. **量化后线性 kernel 的名称变了，还要排除 `lm_head`。** FP8 和 MXFP8 都叫 `device_kernel`（用 `demangledName` 区分），Triton INT8 叫 `scaled_mm_kernel`。`lm_head` 不量化，每次采样仍有一个 gemvx `kernel`，但它在 `Step/*` 之外。
28. **激活量化的 kernel 归类因方案而异。** MXFP8 是独立的 `MXFP8QuantizeSwizzledKernel`，INT8 是独立的 `dynamic_scaled_int8_quant_kernel`，每 step 各 112 个。Triton 融合 kernel 名字带 `mxfp8_quantize`、`dynamic_scaled_int8_quant`，但时间和 bf16 相同，不含这部分计算。FP8 没有独立 kernel，`abs_amax` 融合在 Triton kernel 里，和 norm、silu 拆不开。
29. **kernel 时间有约 0.25～0.28 ms 的离散抬升。** 各路线都有，`lm_head`、线性层合计和注意力计算都见过。总和的中位数可能落在高档，曾让 D 的 bf16 Decode 线性时间在 1.86 和 2.11 ms 间跳动，也让 A 的 FP8、INT8 后处理看起来变慢。线性层要用四类中位数之和，同时看分位数。原因未确认。
30. **`Step` 的 Prefill GPU 跨度只有大批量时可靠。** A、F 的跨度远大于 kernel 时间（A 单次 5.4～13.5 ms，kernel 3～5 ms），且不是单个空隙造成的。这两个配置要看 kernel 时间。
31. **`temperature=0` 的 GSM8K 重复跑，也有约 4.5% 的题目对错不同。** bf16 两次 1319 题里有 59 题不同，只有 867 题文本相同。比较准确率要有 bf16 重复对照，并做配对检验。
32. **中途打断门控请求时，`SIGINT` 可能停不下服务。** 打断时服务处于 `/pause` 附近，`pkill -INT` 后进程一直在，`SIGTERM` 才退出。是否由 pause 引起未确认。
33. **自写脚本不要命名为 `kernels.py`。** `transformers` 导入时会探测 `kernels` 包，脚本目录在 `sys.path` 最前面，会把脚本当成那个包执行。
34. **`compressed_tensors` 0.17.0 的 INT8 预设名是 `W8A8` 或 `INT8`，没有 `INT8_W8A8`。** 两者都是权重 per-channel、激活动态 per-token，都对称。checkpoint 里不写 `weight_zero_point` 也能被 vLLM 加载。
35. **Prefill 不总是 GPU 受限。** kernel 变快后，GPU 会转而等 CPU 发射，这时 GPU 跨度基本等于 `Step` 的 CPU 时间，量化路线明显大于 kernel 时间。A 全部路线和 F 的 MXFP8、INT8 都是这样（如 F 的 MXFP8：CPU 31.8 ms，跨度 31.4 ms，kernel 16.9 ms）。这种配置里 kernel 变快不再降低 TTFT。这解释了坑 30 的现象，CUPTI 对这部分 CPU 时间的放大没有量化。
36. **非流式请求没有客户端 TTFT。** 可以从 nsys 取采样 kernel（`reduce_kernel`、`_gumbel_sample_kernel`）的结束时间算出每个 token 的就绪时刻，得到的是引擎侧数值。本次它比客户端总耗时少 4.5～7.4 ms。
37. **稠密 kernel 不跳过零，权重置零不提速。** 50% 剪枝后的 INT8 和未剪枝版本的 TTFT、TPOT、整体耗时差异在噪声内，模型显存和 KV 容量完全相同。带目标的 `sparsity_config` 会在 `compressed_tensors.py:298` 报错。vLLM 没有稀疏线性 kernel，所以剪枝类论文的提速不能靠 checkpoint 复现。
38. **无补偿的 50% 幅值剪枝会毁掉 Qwen3-0.6B。** GSM8K 从 36.6% 跌到 2.1%，1319 条回答全部循环到 256 token 上限。这是无补偿剪枝的结果，不代表 OBR。
39. **困惑度保住不代表算术能力保住。** OBR 把非结构化 50% 的困惑度控制在 19.57（基线 16.51），GSM8K 仍从 39% 跌到 2.1%。这不是 vLLM 路径问题，HF 加浮点激活跑前 160 题也是 2.5%。评估剪枝不能只看困惑度。
40. **SM120 有原生 2:4 稀疏 Tensor Core，但只有 FP8/FP6/FP4。** 见 CUTLASS 的 `mma_sm120_sparse.hpp`，没有 INT4/INT8。稀疏时 k 为 64，稠密为 32。
41. **`.venv` 有 nvcc 和 CUTLASS，但没有 cmake。** nvcc 在 `nvidia/cu13/bin`，CUTLASS 在 `flashinfer/data/cutlass`（工具头文件在 `tools/util/include`）。要用 `-gencode=arch=compute_120a,code=sm_120a`，`-arch=sm_120` 和 `sm_120f` 都会在 ptxas 报 `.kind::f8f6f4 not supported`。nvcc 13.5 配 13.2 头文件要加 `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`。`-fPIC` 要写成 `-Xcompiler -fPIC`，`libcudart` 只有 `.so.13`，要用 `-Xlinker <路径>`。
42. **稀疏操作数只能是 A，所以权重要当 A。** 这时 token 在 N 维。D 要用 ColumnMajor，既满足 N 不对齐的情况，也和 torch 的 `[token, 输出]` 内存布局一致，x 也不用拷贝。元数据每个 atom 是 128×128，2048 字节，先沿 m 再沿 k，每组 4 位，编码见 `cutlass_sparse.py`。
43. **稀疏 kernel 只在大批量有明显收益。** 1024 token 以上约快 1.3～2.4 倍，256 token 以下耗时只有 25～50 µs 且波动大。所以 Decode（M 为 1 或 8）的 TPOT 没有收益。
44. **`FP8ScaledMMLinearKernel.apply_weights` 用 `w.shape[1]` 推输出宽度。** 换权重布局的 kernel 不能直接用它给的 `output_shape`。
45. **生成多份完整 checkpoint 时不要把浮点权重留在内存里。** 5 份 float32 的线性层权重加 Hessian 超过 15 GB，进程被杀，日志没有报错，只是最后一行停住。生成时就压成 1 字节。
46. **会话临时目录跨天会被清空。** 这次的脚本都没了。要保留的脚本放到 `/home/bca/models/obr/scripts/`。
47. **`quant_method` 不只在线性层上。** Qwen3 的 `Attention` 模块也有这个属性，但没有 `weight`。按 `quant_method` 找线性层时要同时判断 `weight`，否则 eager 下启动就抛 `AttributeError`。
48. **SM80 的 2:4 元数据有行重排，CUTLASS 只提了一句。** 每个 uint32 存 8 组、每组 4 位、低位在前，但内存位置是 `word = (w//2)*2M + (m//16)*32 + (m%8)*4 + (m%16)//8 + (w%2)*2`（`w = g//8`）。这是在 3090 上逐 nibble 扫描测出来的，2048 项一一对应。全相同的掩码在错误布局下也算得对，只有随机掩码能暴露问题，症状是相对误差约 1.07。
49. **SM80 稀疏 GEMM 的 N 必须对齐。** N=1、17 实测使设备出错，之后的 cuBLAS 调用才报 `CUBLAS_STATUS_ALLOC_FAILED` 或 `EXECUTION_FAILED`，看起来像 cuBLAS 的问题。kernel 里把 N 补到 16 的倍数再裁掉。另外 M 要被 16 整除、K 要被 128 整除。
50. **SM80 稀疏 GEMM 的 D 只能行主序。** 输出是 `[M, N]`，要转置成 `[N, M]`。这次转置约占稀疏 GEMM 时间的 49%，足以抵消全部提速。不要只看 GEMM kernel 就下结论。
51. **稀疏 MMA 要求 Warp 层的 k 是指令 k 的偶数倍。** 指令是 k64，所以 `WarpShape` 的 k 要 128，Tile 的 k 也至少 128。include 顺序也要和 `scaled_mm_c2x.cuh` 一致，否则 `visitors.hpp` 报 `gemm is not a type name`。
52. **服务器上的基线只在暂存区时，不要跑 `git checkout -- .`。** 它把工作区恢复成基线，撤销所有补丁。要重放的话先删掉补丁新增的文件。
53. **远程命令里不要用裸的 `pkill -f <命令片段>`。** 它会匹配到当前 shell 自己的命令行，把连接或脚本一起杀掉。用 `[v]llm` 这种写法，或把命令写进脚本再触发。
54. **`hf download --revision <sha>` 不写 `refs/main`。** 之后 `HF_HUB_OFFLINE=1` 用模型名启动会报 `LocalEntryNotFoundError`。手动写入 `refs/main`。
55. **这台租用机（gpuhome.cc）的网络特点。** 本机到服务器上行只有约 3KB/s，不能传大文件或本地下载模型再上传。GitHub 直连约 30KB/s，`gh-proxy.com` 约 2.8MB/s。`hf-mirror.com` 单文件有时只有 1MB/s 以下，ModelScope 单连接约 1.2MB/s，16 路并行约 8MB/s。第三方来源的代码要比对 Git tree hash，模型要比对 LFS sha256。
56. **按列主序直接写输出不等于省掉转置开销。** 自定义 store 让 8192 token 的 GEMM 从 370 ms 涨到 490 ms。每个线程持有的 4 个元素在 n 方向，列主序下相隔 M，只能逐元素写，写入无法合并。它比“GEMM + torch 转置”的 552 ms 好，但仍达不到 1.3 倍。
57. **Qwen3-8B 上 2:4 比 0.6B 更不可用。** 稠密 INT8 的 GSM8K 是 88.25%，INT8 + 2:4 不管有没有 OBR 都只有 1%～2%。0.6B 的低基线没有掩盖什么，崩溃是一样的。
58. **8B 上 OBR 的困惑度比无补偿更差，原因未查明。** 1749 对 748，而单层目标函数确实下降了。不要直接引用这个 OBR 结果代表论文方法。困惑度只有 bf16 和 INT8 路线有可信的参照。
59. **8B 的 OBR 用批量 PCG 只要 22 分钟。** 单块 36 秒，Cholesky 外推约 5 小时。`down_proj` 的 Hessian 是 604MB，单卡够用。
60. **困惑度可以直接在 `vllm serve` 里算。** `/v1/completions` 带 `prompt_logprobs: 0`、`max_tokens: 1`，第一项是 None 要跳过。`--max-model-len 2048` 时每段最多 2047 token。这条路径带激活量化，和 HF 浮点激活的旧数据不是同一口径。
61. **更正第 57、58 条：8B 的 OBR 第一版有符号错误。** 剪掉部分的误差应为 `-w`，写成了 `+w`，补偿方向相反。修复后 INT8 + 2:4 + OBR 的 GSM8K 是 44.43%，困惑度 11.97，不是 1%～2%。和直接解对比目标函数检查不出这种错，因为两边用的是同一个误差向量。要用“补偿后误差 < 只剪枝误差”来断言。
62. **查这类问题先比校准集上的层输出误差。** 校准集上比无补偿还差，就是实现错，不用再怀疑过拟合或激活量化。脚本在 `profiles/nsys/phase_cost/quant/obr_3090/scripts/obr_diag.py`。
63. **torch profiler 能看到 CUDA Graph 里的 kernel。** 和 nsys 默认的整图模式（第 17 条）不同，Kineto 把图内 kernel 逐个记下来，用 `cudaGraphLaunch` 的 `correlation` 关联。3090 上 Decode 全部 32/32、128/128 步走图，每层 kernel 都在。
64. **`vllm serve` 的前端（AsyncLLM）trace 是空的。** `with_stack=false` 时文件只有 858 字节、0 个事件，`output_handler` 里加的 `record_function` 也没有记下。推测是因为前端 profiler 在 `asyncio.to_thread` 的线程里启动，没有验证。HTTP、detokenize 只能用时间差推算。
65. **trace 的 `ts`（µs）加 `baseTimeNanoseconds/1000` 就是墙钟。** 和同一台机器上客户端的 `time.time_ns()` 能直接对齐。实测从 `/resume` 到第一次 `schedule` 是 1.1～2.4 ms，没有出现负值。
66. **V2 runner 和 EngineCore 原本没有 profiler 作用域。** `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 只打开调度器里的 3 个作用域，看不出 schedule、sample_tokens、输入准备各花多少。CHG-017 加了 `core: *`、`runner: prepare_inputs`、`frontend: process_outputs`，同样受这个环境变量控制。
67. **serve 和 `vllm bench forward` 的非线性 kernel 不一样。** serve 里 norm、rotary、激活是 Inductor 生成的 Triton kernel，bench forward 用的是 `vllm::rms_norm_kernel` 等自定义算子，两边的非线性部分不能逐项比。Triton kernel 的名字会带上所有融合进去的算子，例如 `triton_red_fused_dynamic_scaled_int8_quant_fused_add_rms_norm`，但真正的 INT8 量化仍是独立的 `vllm::dynamic_scaled_int8_quant_kernel`，每层 4 个。按子串 `int8_quant` 归类会把 norm 算成量化。
68. **torch profiler 会放大 CPU 受限的 Prefill。** INT8 的 A：开 profiler 时 TTFT 39～50 ms，不开时 31.6 ms（CHG-015）。bf16 的 A 是 GPU 受限，只多约 1 ms。TPOT 的差异都在 1% 以内。
69. **3090 实测上限和白皮书对不上，原因是时钟和功耗。** bf16 负载下 SM 时钟是 1935 MHz，白皮书按 1695 MHz 计算，所以实测 77.8 TFLOPS 高于 71。INT8 撞上 350 W 功耗墙（throttle reason 0x4），时钟降到 1425～1650 MHz，实测只有 211～235 TOPS，是 284 的 74%～83%。显存时钟报 9501 MHz，即 19.0 Gbps，理论 912 GB/s，实测读带宽 887 GB/s。
70. **`*.kernels.json` 按 kernel 名合并，分不出 qkv、o、gate_up、down。** 要分投影，得读原始 trace，按层内顺序归类（第 19 条），脚本在 `obr_3090/scripts/roofline_kernels.py`。注意 `reshape_and_cache_flash_kernel` 的名字里有 `flash`，要先于注意力判断。
71. **刚写完的小张量会留在 L2，按显存字节算出的带宽可能超过 100%。** 例如 t256 的 `down` 激活量化，输入 6.3 MB，L2 是 6 MB，算出来是 109%。小形状的“带宽占比”只能当上限参考。
72. **8B 上 2:4 的精度主要被校准数据卡住。** 把 WikiText 换成模型自己按评测格式解 GSM8K 训练题生成的文本后，INT8 + 2:4 + OBR 的 GSM8K 从 44.43% 升到 81.20%，INT4 从 19.56% 升到 69.90%。代价是 WikiText 困惑度从 11.97 升到 21.24。校准数据的领域决定了补偿保住哪种能力，评估要同时看目标任务和通用困惑度。
73. **自生成校准的窗口开头就是评测用的 5 道例题（`train[:5]`）。** 校准分布和评测提示词的前缀完全相同，题目本身不同（训练集对测试集）。这让 GSM8K 结果偏乐观，换其他任务时不能期望同样的提升。
74. **混合校准能兼顾两头。** 64 个 WikiText 窗口加 64 个自生成窗口：INT8 + 2:4 + OBR 的 GSM8K 77.41%，困惑度 12.85，比纯 WikiText 版只高 0.88，GSM8K 比纯自生成版只低 3.79 个百分点。追求 GSM8K 用纯自生成，追求通用性用混合。

72. **`pgrep -f "[s]tress_bf16.py"` 照样会匹配到自己。** 坑 53 的 `[x]` 写法只在命令行里没有原样文件名时才管用。同一条远程命令里还有 `gunzip > stress_bf16.py`，`pgrep` 把当前 shell 也匹配了，`kill` 掉了 ssh 会话，留下一个孤儿 `vllm serve`。长任务用脚本文件启动，记下 pid 或进程组，按 pid 杀（`kill -TERM -<pgid>`）。
73. **`/proc/<pid>/stat` 的进程名只有 15 个字符。** EngineCore 显示为 `VLLM::EngineCor`，按 `EngineCore` 匹配会找不到，导致把自己的 EngineCore 当成外来的 GPU 进程。
74. **`--max-num-batched-tokens` 会改变 KV 容量。** 同样是 `--gpu-memory-utilization 0.85`，bf16 Qwen3-8B 在 8192 时 KV 是 23,552 token，2048 时是 20,480 token，原因未查。比较分块大小时要同时报告 KV 容量。
75. **闭环压测加固定长度，请求会成波次到达。** 一波同时结束，下一波同时到达，整波一起 Prefill，TTFT 和最大 ITL 偏大（并发 16 时 TTFT P50 1.71 s，就是一次 8192 token 的 Prefill）。看 TTFT 和 SLO 要用 Poisson 到达的结果。
76. **`--max-num-batched-tokens` 不小于 prompt 长度时，单个 Prefill 不会被切开。** 1024 token 的 prompt 在 2048 下仍是一整块，Decode 照样被卡约 235～260 ms。要减少卡顿，块要小于 prompt。
## 设计决定

| 决定 | 原因 |
| ---- | ---- |
| 用 `BatchReqState.is_prefilling_np` 判定阶段，不用 `num_computed < num_prompt_tokens` | `prefill_len` 包含抢占恢复时已生成的 output。实测反例：`computed=64, prompt=40, prefill_len=96`，请求在重算 output，用 prompt 长度会误判成 Decode |
| 在运行时包裹 `layers.N.self_attn/mlp` 的 `forward`，不改模型文件 | 关闭时层内零判断；模型计算代码零改动；结构相同的模型（如 Qwen3 MoE）无需改动 |
| 在 `Worker.compile_or_warm_up_model` 之后安装，不在 `load_model` | 见坑 2。warmup 期间不存在包裹，也就不可能被打标签 |
| `enable_nvtx_ranges` 用 `is no_step_range` 判断是否已安装 | `elastic_execute.py` 会重入 `compile_or_warm_up_model`，重复包裹会让 range 嵌套两层 |
| 阶段放在 `_PhaseRanges.phase`，不进 forward context | V2 的 FULL 分支不经过 `set_forward_context`；DBO 微批的 context 是另建的，要额外透传。写入方只有 `execute_model` 所在线程 |
| 非 eager 时保留 `Step/*`，不改编译配置 | 用户明确要求不擅自改配置。Step 在 compiled 图外，仍然准确 |
| V1 runner 只打 warning | Qwen3 默认不走 V1，避免维护两条路径 |
| 文档放 `.agents/research/` | `docs/contributing/*` 会被 glob 进文档站，这里是内部交接记录 |

## 改动记录

追加模板：

```markdown
### CHG-NNN：一句话标题

- 日期 / 作者：
- 文件：
- 内容：
- 原因：
- 验证：实际跑了什么、结果如何；没跑的也写明
- 状态：未提交 / 已提交（hash）
```

### CHG-001：初版分层 NVTX 插桩

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`vllm/utils/nvtx_phase_ranges.py`（新增）、`vllm/envs.py`、`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu_worker.py`、`docs/contributing/profiling.md`
- 内容：
  - 新增 `classify_phase`、`no_step_range`、`_PhaseRanges`、`install_nvtx_ranges`。
  - `envs.py` 注册 `VLLM_NVTX_ENABLE`，默认关闭。
  - `execute_model` 用 `with self.nvtx_step_range(batch_req_state):` 包住前向的三条分支（FULL / PIECEWISE / eager）。这段 diff 大多是重新缩进，用 `git diff -w` 看。
  - `GPUModelRunner` 增加类属性 `nvtx_step_range`。放在类上是因为测试里有 `GPUModelRunner.__new__` 的用法。
  - `_make_model_runner` 在走 V1 时打 warning。
  - `profiling.md` 增加 Per-layer NVTX ranges 小节。
- 原因：用户需要按 Prefill / Decode / Mixed 分层查看 Attention 和 FFN 耗时。
- 验证：初版的 nsys 对账发现了 CHG-003 的问题，结果作废。最终验证见「验证结果」。
- 状态：未提交

### CHG-002：简化非 eager 的 warning

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`vllm/utils/nvtx_phase_ranges.py`
- 内容：warning 不再打印编译模式和 Graph 模式的数值。
- 原因：`CompilationMode` 是 `IntEnum`，打印出来是 `compilation mode 3`，没有可读性。
- 验证：非 eager 运行的日志里 warning 文本正确。
- 状态：未提交

### CHG-003：修复 warmup 污染 NVTX 统计

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu_worker.py`、`vllm/utils/nvtx_phase_ranges.py`
- 内容：
  - 安装时机从 `load_model` 移到 warmup 之后。
  - 新增 `GPUModelRunner.enable_nvtx_ranges`，并让它可重入。
  - `Worker.compile_or_warm_up_model` 的 V2 分支在 `eplb.suppress()` 结束后调用它。
- 原因：坑 2。对账时四个场景都恰好多 6 个 Step，去掉开头 6 个后序列与调度记录完全一致，说明判定正确，只是 warmup 混了进来。
- 验证：修复后 5 个场景的 Step 序列全部逐项一致。
- 状态：未提交

### CHG-004：恢复 `capture_model` 的装饰器

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`vllm/v1/worker/gpu/model_runner.py`
- 内容：把 `enable_nvtx_ranges` 移到 `@torch.inference_mode()` 之上，让装饰器回到 `capture_model`。
- 原因：CHG-003 插入位置错误，夹在装饰器和 `capture_model` 之间，`capture_model` 失去了 `inference_mode`。eager 不调用它，测试没有暴露，是读 diff 时发现的。
- 验证：重跑非 eager 的 `mixed`（40/40）和 eager 的 `mixed`（40/40），eager 的开/关各一次输出与基线一致。
- 状态：未提交

### CHG-005：新增本交接文档

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（新增）
- 内容：记录状态、约束、已知坑、设计决定、改动历史和验证方法。
- 原因：用户要求记录每次改动，交付给其他 agent 阅读。
- 验证：已用脚本核对文中引用的文件和符号都存在。
- 状态：未提交

### CHG-006：记录 Prefill / Decode 开销测量和新发现的坑

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（本文件）。源码未改动。
- 内容：
  - 「不要重新踩的坑」新增第 10 至 16 条。
  - 「当前状态」新增「最近一次测量」一行。
  - 测量产物放在 `profiles/nsys/phase_cost/`，完整结果和实际命令见其中的 `report.md`。不合格的试跑留在 `rejected/` 作证据。
- 原因：用户要求把这次测量中发现的坑追加到本文件。
- 验证：
  - Qwen3-0.6B 四个配置（256×1、1024×1、256×8、1024×8，`max_tokens=33`，窗口内重复 5 次）各跑 2 轮，共 8 个报告。
  - 8 个报告都是 Prefill 5、Decode 160、无 Mixed，序列逐项正确，28 层齐全，各层标签数等于该阶段 Step 数。
  - `execute_context` 标签独立确认批内请求数和 token 数。
  - 逐 step 还原与 `nsys stats` 对账一致：Step 的 CPU 和投影中位数、抽查的单层投影中位数，以及 A_r1、D_r1 的 kernel 总时间。
  - 未验证：CUPTI 追踪对 CPU 时间的影响；Decode 慢 step（9%～14% 超过中位数 1.3 倍）的原因；256→1024 配置中 r2 的 kernel 尖峰原因；`Layer_00` 在 CPU 侧稳定偏高的原因。warmup 请求之后它仍高出 26%～35%，所以坑 7 的冷启动推断不足以解释这部分。
  - `ruff`、`pre-commit`、`pytest` 仍未运行，本次只改了文档。
- 状态：未提交

### CHG-007：记录非 eager 基线和量化可行性

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（本文件）。源码未改动。
- 内容：
  - 「不要重新踩的坑」新增第 17 至 24 条。
  - 「必须遵守的约束」增加量化对比测量的例外。
  - 「当前状态」更新「最近一次测量」。
  - 更正 CHG-006 的未验证项：CUPTI 的影响已测，eager 放大约 1.3～1.5 倍，非 eager 基本没有影响。
  - 更正“Decode 受 CPU 限制”：这只在 eager 下成立。非 eager 下一次请求里 GPU 有 94%～98% 的时间在忙，Decode 周期里 GPU 只空闲约 0.08 ms。
- 原因：用户要求先测非 eager 基线，再准备量化对比。
- 结果（非 eager，33 token，不带 nsys，每配置 15 次）：
  - 请求耗时：256×1 为 100 ms，1024×1 为 121 ms，256×8 为 160 ms，1024×8 为 317 ms。eager 分别是 325、317、344、463 ms。
  - 模型线性层占请求时间 53%～55%，加上约 0.5 ms/step 的后处理共 60%～72%。
  - 注意力计算（含 KV 读取）占 9%～26%，KV 写入不超过 1.6%。
  - 编译几乎不改变 kernel 时间，节省的是 CPU 时间。
- 量化可行性（单次矩阵乘，1024→6144，随机数据，不含激活量化）：
  - M=8192：bf16 1626 µs，CUTLASS FP8 862 µs，Triton INT8 707 µs。
  - M=1024：bf16 252 µs，CUTLASS FP8 174 µs，Triton INT8 128 µs。
  - M=256 的结果不单调（Triton 166 µs、FP8 147 µs、bf16 70 µs），小批量不能下结论。
  - `mxfp4` 所需的 FlashInfer CuteDSL 可用，`cutlass_scaled_mm_supports_fp4(120)` 为 True。未在模型上运行。
- 验证：
  - 12 个非 eager nsys 报告都是 Prefill 5、Decode 160、无 Mixed，每 step 线性 kernel 都恰好 112 个。
  - 不带 nsys 的 8 次运行全部完成。部分日志在 SIGINT 关闭时有 `EngineDeadError`，出现在全部请求返回 200 之后，原因未分析。
  - 未验证：任何量化方案在 Qwen3 上的实际运行、性能和精度；未归因的 14～20 ms 的构成；KV 读取和注意力计算的拆分。
  - `lm_eval` 和 `datasets` 不在 `.venv` 里。`ruff`、`pre-commit`、`pytest` 仍未运行。
- 状态：未提交

### CHG-008：记录 bf16 与量化方案的对比

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（本文件）。源码未改动。
- 内容：
  - 「不要重新踩的坑」新增第 25 至 34 条。「当前状态」更新「最近一次测量」。
  - 测量产物在 `profiles/nsys/phase_cost/quant/`，完整结果和实际命令见 `report.md`。
  - INT8 checkpoint 在 `/home/bca/models/qwen3-0.6b-int8-w8a8-dynamic/`，不进 Git。
  - 更正 CHG-007：`mxfp4` 所需组件可用，但在模型上运行会失败，见坑 25。
- 原因：用户要求比较 bf16 与量化方案的性能和精度。用户中途要求只用 nsys、减少组合、总时长控制在 1 小时内，所以只测 A、D、F。
- 结果（请求耗时相对同配置首尾两次 bf16）：
  - FP8：A −11%～−16%，D −20%～−25%，F −23%。
  - INT8：A −6%～−11%，D −27%～−31%，F −18%～−19%。
  - MXFP8（关闭 autotune）：A +13%～+20%，D −11%～−17%，F 在噪声内。
  - MXFP4：启动失败。
  - 模型显存 1.12→0.71～0.73 GiB，KV 容量 +7%～+13%。
  - GSM8K（1319 题）：bf16 39.04%、39.58%，FP8 37.76%，MXFP8（默认 autotune）37.00%，INT8 36.62%。Bonferroni 校正后都不显著。INT8 触顶 256 token 的回答为 123 条，bf16 为 35～41 条。
- 验证：
  - 15 个 nsys 报告全部通过：Prefill 5，Decode 为 5×(max_tokens−1)，无 Mixed，`execute_context` 与批内容一致，每 step 线性 kernel 恰好 112 个。
  - 未验证：配置 B、C、E；默认 autotune 的 MXFP8 的 kernel 构成（只对照了请求耗时）；其 tactic；MXFP4 的其他 linear backend；MXFP8 Decode 慢的原因；INT8 触顶增加的原因；坑 29、30 的原因。
  - 耗时结论基于每组合 5 次，不是原计划的 2 轮 15 次。不带 nsys 的数据只有部分。
  - `ruff`、`pre-commit`、`pytest` 仍未运行，本次只改了文档。
- 状态：未提交

### CHG-009：从 nsys trace 计算 TTFT 和 TPOT

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（本文件）。源码未改动。
- 内容：
  - 「不要重新踩的坑」新增第 35、36 条。第 35 条解释了 CHG-008 记录为未确认的坑 30。
  - `profiles/nsys/phase_cost/quant/report.md` 新增第 8 节，数据在 `nsys/ttft_tpot.json`。没有新增采集。
- 原因：用户要求计算 TTFT 和 TFOP，这里把 TFOP 理解为 TPOT。
- 结果（相对首尾两次 bf16）：
  - TPOT：FP8 A −18%～−20%、D −12%～−18%、F −22%。INT8 A −14%～−16%、D −16%～−21%、F −18%。MXFP8 A +15%～+18%、D +2%～+9%、F +1%。
  - TTFT：D 中 FP8 −29%～−34%、MXFP8 −32%～−37%、INT8 −38%～−42%。F 中 FP8 −36%、INT8 −27%～−28%、MXFP8 −5%～−6%。A 不下结论。
- 验证：
  - 引擎侧总耗时比客户端请求耗时少 4.5～7.4 ms，15 个运行一致。TPOT 与 Decode 周期相差不超过 2%。
  - 未验证：流式客户端的 TTFT；不带 nsys 时的 TTFT；默认 autotune 的 MXFP8；量化路线 Prefill CPU 时间增加的原因。
- 状态：未提交

### CHG-010：评估 OBR（量化 + 剪枝）在 vLLM 上的可行性

- 日期 / 作者：2026-11-06 / Claude Code
- 文件：`.agents/research/nvtx-profiling-changelog.md`（本文件）。源码未改动。
- 内容：
  - 「不要重新踩的坑」新增第 37、38 条。
  - 报告在 `profiles/nsys/phase_cost/quant/sparsity/report.md`。
  - 剪枝 checkpoint 在 `/home/bca/models/qwen3-0.6b-int8-w8a8-dynamic-sp50/`，不进 Git。
- 原因：用户询问能否参考 OBR 论文做量化加剪枝并给出性能报告。这是「必须遵守的约束」之外的新请求，只用了仓库外的 checkpoint。
- 结果：
  - 不改代码无法用 OBR 得到 TTFT、TPOT 提升。OBR 本身没有运行，所以没有 OBR 的性能数据。
  - 流式实测（不带 nsys，2 轮，每组合 15 次），整体耗时相对 bf16：FP8 A −15.5%、D −25%、F −22%～−23%；INT8 A −11%～−13%、D −26%～−27%、F −20%～−21%；INT8 + 50% 剪枝与 INT8 差异在噪声内。
  - TPOT 相对 bf16：FP8 −17%～−23%，INT8 −15%～−20%。
  - TTFT 相对 bf16：D、F 降 29%～37%，A 无收益。
  - 更正 CHG-009 的未验证项：不带 nsys 时 A 的 TTFT 同样没有因量化而降低，CPU 受限不是 nsys 造成的假象。
- 验证：
  - 8 次流式运行和 1 次 GSM8K 全部完成，usage 校验通过，剪枝权重零值占比 50.0%～50.1%。
  - 未验证：OBR 在 Qwen3-0.6B 上的精度和校准耗时；W4A16；4-bit KV cache；论文的提速数字和 2:4 kernel。论文内容来自页面读取，未逐表核对。
  - `ruff`、`pre-commit`、`pytest` 仍未运行，本次只改了文档。
- 状态：已提交（`5177252a24`，用户提交）

### CHG-011：复现 OBR 精度并新增 2:4 稀疏 FP8 kernel

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：见「当前状态」的 CHG-011 文件列表，以及本文件。
- 内容：
  - 新增 `--linear-backend cutlass_sparse`：SM120 的 2:4 稀疏 FP8 linear kernel。默认选择不受影响，放在 `_POSSIBLE_FP8_KERNELS` 末尾，只有显式指定才会用。
  - 新增 `tools/build_sparse_fp8_sm120.sh`，用 `.venv` 的 nvcc 编译 `vllm/_sparse_fp8_sm120.so`，运行时用 `ctypes` 加载。未接入 CMake，因为环境没有 `cmake`。
  - 「不要重新踩的坑」新增第 39 至 46 条，「必须遵守的约束」增加例外。
  - 报告在 `profiles/nsys/phase_cost/quant/obr/report.md`。checkpoint、Hessian 和脚本在 `/home/bca/models/obr/`，不进 Git。
- 原因：用户要求复现 OBR 精度并批准写 kernel。硬件没有 INT4 稀疏 MMA，所以实现的是 FP8。
- 结果：
  - GSM8K：bf16 39.04%、39.58%，稠密 FP8 38.59%。2:4 + OBR 为 0.91%～1.74%，级联校准后 1.36%，只剪 FFN 2.73%，非结构化 50% 2.12%。无补偿 0.99%。
  - 困惑度：基线 16.51，稠密 FP8 16.65，无补偿 93.84，2:4 + OBR 37.46，加级联 30.92，只剪 FFN 22.32，非结构化 19.57。
  - 稀疏 vs 稠密 FP8（同一 checkpoint，流式，2 轮）：TTFT −19%～−28%，整体耗时 −2%～−12%，TPOT 在噪声内。
  - 微基准（空闲 GPU）：相对稠密 FP8，1024 token 以上为 0.41～0.75 倍，256 token 以下不下结论。
- 验证：
  - `compress_24` 和 kernel：5 种形状、批量和单值 scale，相对误差 0.17%，非 2:4 权重被拒绝。
  - 端到端：`Selected CutlassSparseFP8ScaledMMLinearKernel`，GSM8K 1.67%（同一 checkpoint 的稠密 kernel 1.36%）。
  - 未验证：加载后的实际显存；E2M1、FP6；其他 GPU；A 上 TTFT 下降的原因；论文的 5.9 倍；OBR 官方实现的对照。`ruff`、`pre-commit`、`pytest` 仍未运行，没有写仓库测试。
- 状态：未提交

### CHG-012：NVTX 标签加入步序号、批信息、硬件和线性 kernel

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`vllm/utils/nvtx_phase_ranges.py`、`vllm/v1/worker/gpu/model_runner.py`、`docs/contributing/profiling.md`、本文件。
- 内容：
  - `Step/<phase>` 后面加 `#<步序号> reqs= tokens= cudagraph= hw= linear=`，非 eager 也有。
  - eager 下新增 `<phase>/Layer_NN/<Attention|FFN>/<线性层> kernel= shape=`。`shape` 是加载后存储的权重形状，FP8 是 `[K, N]`，稀疏 kernel 压缩后 K 减半。
  - 「不要重新踩的坑」新增第 47 条。修正「当前状态」的提交状态：CHG-011 已随 `e26b649a68` 提交。
- 原因：用户要求 3090 实验直接在 vLLM 内测量，标签带上 step、hw、kernel。
- 影响：`Step/*` 标签现在每步唯一，统计要按前缀匹配，不能等值匹配。这之后的 trace 不能直接套用旧的 `label = 'Step/Prefill'` 查询。
- 验证：
  - 本地 RTX 5070 + nsys，Qwen3-0.6B。非 eager 的 bf16：5 个 Step，`cudagraph` 为 Prefill 0、Decode 1。eager 的 `fp8_seq_none`：`linear=CutlassFP8*112,UnquantizedEmbedding*2`，6 类标签各 140 条（28 层 × 5 步），生成 token 数正确。
  - 未验证：RTX 3090 上的标签；`vllm bench forward` 是否带这些标签；开启后的开销（新标签每步多一次字符串格式化）；多 GPU。`ruff`、`pre-commit`、`pytest` 未运行，没有写仓库测试。
- 状态：已提交并 push

### CHG-013：新增 SM86 的 2:4 稀疏 INT8 kernel，并在 RTX 3090 上用 vLLM 测速

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`csrc/quantization/w8a8/cutlass/sparse_int8_mm_sm80.cu`（新增）、`tools/build_sparse_int8_sm80.sh`（新增）、`vllm/model_executor/kernels/linear/scaled_mm/cutlass_sparse.py`、`vllm/model_executor/kernels/linear/__init__.py`、`vllm/config/kernel.py`、本文件。结果在 `profiles/nsys/phase_cost/quant/obr_3090/forward/`。
- 内容：
  - `--linear-backend cutlass_sparse` 在 SM86 上选 `CutlassSparseInt8ScaledMMLinearKernel`，放在 `_POSSIBLE_INT8_KERNELS` 末尾，默认路径不变。要求动态对称激活，M 被 16、K 被 128 整除，非 2:4 权重加载时报错。
  - 「不要重新踩的坑」新增第 48 至 55 条。
  - 用户把计划改成全部在 vLLM 内测量：取消仓库外微基准，改用 `vllm bench forward --profile-dir`。OBR checkpoint 的生成仍要在 vLLM 外。
- 原因：用户要求在 RTX 3090 上验证 OBR 的 2:4 kernel。
- 结果（Qwen3-8B，`vllm bench forward`，200 次回放中位数，2:4 无补偿 checkpoint）：
  - ≥1024 token：稀疏 GEMM 相对稠密 INT8 快 1.39～1.42 倍，加上输出转置后为 0.95～0.99 倍，整个前向 0.97～0.99 倍。**未达到 1.3 倍的通过标准。**
  - 256 token 及以下，线性部分 1.06～1.39 倍，Decode 的 d1、d8 为 1.24、1.19 倍，整个前向 1.16、1.11 倍。
  - 相对 bf16，8192 token：稠密 INT8 GEMM 快 2.9 倍，稀疏 GEMM 4.2 倍，加转置 2.8 倍。
  - 权重零值不影响稠密 kernel：`int8` 和 `int8_24` 的差异约 1.2%。
- 验证：
  - 数值：5 种真实形状 × 9 种 N 加单值 scale，和浮点参考的相对误差约 0.17%。元数据映射公式和实测表 2048 项全部一致。
  - 端到端：`Selected CutlassSparseInt8ScaledMMLinearKernel`，bench 的 `finite` 和 `graph_deterministic` 为真。
  - 未验证：真实权重下稀疏和稠密 kernel 的输出是否一致；`vllm serve` 下的稀疏路线；INT4；OBR；GSM8K；端到端 TTFT、TPOT。`ruff`、`pre-commit`、`pytest` 未运行，没有写仓库测试。
- 状态：已提交并 push

### CHG-014：把输出转置融合进 SM86 稀疏 INT8 kernel

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`csrc/quantization/w8a8/cutlass/sparse_int8_mm_sm80.cu`、`vllm/model_executor/kernels/linear/scaled_mm/cutlass_sparse.py`、本文件。结果在 `profiles/nsys/phase_cost/quant/obr_3090/forward/`（`sparse_col.json`、`summary_col.txt`）。
- 内容：
  - 新增 `ColumnMajorStore`，kernel 直接写 `[N, M]`，不再调用 torch 转置，补齐后的裁剪也只是视图。
  - 「不要重新踩的坑」新增第 56 条。
  - 用户要求以后每组合只测 3 次。本次重测 `--iters 3 --samples 1`，kernel 分解同样是 3 次回放。
- 原因：用户要求把转置融合进 kernel。
- 结果（对比 `int8_24`，同一 checkpoint；稠密数据是之前 200 次的中位数）：
  - ≥1024 token：线性部分 1.02～1.08 倍，整个前向 1.02～1.05 倍。**仍未达到 1.3 倍。**
  - t8192：GEMM 489.99 ms，稠密 527.14 ms，上一版 GEMM + 转置 552.55 ms。
  - t64 线性 1.42 倍，t1、t8 为 1.27、1.25 倍，d1、d8 为 1.26、1.23 倍，整个前向 1.13～1.35 倍。
- 验证：
  - 5 种形状 × 7 种 N 加单值 scale，失败 0，相对误差约 0.17%。
  - 未验证：真实权重下和稠密 kernel 的输出是否一致；`vllm serve`；端到端。只有 3 次回放，小差异不能下结论。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-015：Qwen3-8B 的 OBR 精度和端到端测量（RTX 3090）

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/`（`report.md`、`eval/`、`scripts/`、`logs/`）、本文件。vLLM 源码未改动。
- 内容：
  - OBR 脚本放在 `scripts/obr_8b.py`：模型在 CPU，逐块上 GPU，批量 PCG，每块一个分片，可续跑。续跑逻辑未实测。
  - 评测放在 `scripts/eval_route.py`：一次 `vllm serve` 内依次测 A、D、F（每组合 3 次）、困惑度和 GSM8K。
  - 「不要重新踩的坑」新增第 57 至 60 条。
- 原因：用户要求回答 8B 上 2:4 + OBR 的 GSM8K 能保住多少。
- 结果：
  - GSM8K：bf16 88.55%，INT8 稠密 88.25%，无补偿 1.82%，INT8 OBR 0.83%（稀疏 kernel 1.52%），INT4 OBR 1.67%。
  - 困惑度：bf16 9.14，INT8 9.13，无补偿 748，INT8 OBR 1749（稀疏 kernel 1797），INT4 OBR 5321。
  - 稀疏 vs 稠密 kernel，同一 checkpoint，整体耗时 1.07～1.15 倍，TPOT 1.10～1.16 倍。稠密 INT8 vs bf16 整体 1.65～2.01 倍。
  - 未达到精度通过标准，也未达到 1.3 倍速度标准。
- 验证：
  - 六条路线都跑完，`Selected` 日志确认了 kernel。A、D、F 的 usage 断言通过。
  - 未验证：OBR 困惑度比无补偿更差的原因；真实权重下稀疏和稠密 kernel 的逐位一致性；60 次迭代下 PCG 和精确解的差距；手写逐块前向和 HF 整体前向相差 1.58% 的原因。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-016：修复 8B OBR 的符号错误，更正 CHG-015 的精度结论

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/`（`report.md`、`eval/`、`eval_signbug/`、`scripts/obr_8b.py`、`scripts/obr_diag.py`、`scripts/decode_ckpt.py`、`logs/`）、本文件。vLLM 源码未改动。
- 内容：
  - `obr_8b.py` 第一步补偿的误差从 `w * (1 - mask)` 改为 `-w * (1 - mask)`。`--check` 增加真实误差断言。
  - 更正 CHG-015 和坑 57、58：它们的 OBR 结论作废。符号错误版本的结果移到 `eval_signbug/`。
  - 「不要重新踩的坑」新增第 61、62 条。
- 原因：用户要求查明 OBR 困惑度比无补偿更差的原因。
- 结果：
  - 修复后 GSM8K：INT8 + 2:4 + OBR 44.43%（稀疏 kernel 45.49%），INT4 + 2:4 + OBR 19.56%。困惑度 11.97、11.99、14.14。
  - 仍未达到“相对 bf16 下降不超过 5 个百分点”。
  - 稀疏 kernel 在真实权重上和稠密 kernel 一致。
- 验证：
  - 第 1 块逐层断言通过，补偿后误差是只剪枝的 23%～44%。36 块和 3 条评测都跑完。
  - 未验证：修复后的端到端测时（速度和权重数值无关，沿用之前的数据）；60 次迭代下 PCG 和精确解的差距。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-017：Qwen3-8B 在 RTX 3090 上的 roofline 和 serving 开销拆解，新增 EngineCore 和 V2 runner 的 profiler 作用域

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：
  - `vllm/v1/engine/core.py`：`step_with_batch_queue` 加 `core: schedule / execute_model / sample_tokens / wait_output / update_from_output`。
  - `vllm/v1/worker/gpu/model_runner.py`：加 `runner: prepare_inputs`（包住 `prepare_inputs` 和 `prepare_attn`）。
  - `vllm/v1/engine/async_llm.py`：加 `frontend: process_outputs`。
  - 新增脚本 `profiles/nsys/phase_cost/quant/obr_3090/scripts/{roof,roofline_kernels,roofline_plot,prof_serve,serve_breakdown,serve_tables}.py`。
  - 结果在 `profiles/nsys/phase_cost/quant/obr_3090/profiling/`（`report.md` 和小的 CSV、JSON、SVG）。
- 内容：
  - 作用域都用已有的 `record_function_or_nullcontext`，只有 `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` 时生效，默认是 `nullcontext`。不改调度和计算逻辑。`step()`（非批队列路径）没有加。
  - 服务器上用 `patch -p1 -b -z .orig_scopes` 打了同样的补丁，原文件备份为 `*.orig_scopes`。
  - 「不要重新踩的坑」新增第 63 至 71 条。
- 原因：用户要求只用 torch profiler 拆解 Qwen3-8B 的瓶颈、开销和 roofline。不开 stack 时，V2 路径上看不到调度、采样和输入准备的 CPU 时间（坑 66）。
- 结果（详见报告）：
  - 实测上限：bf16 77.8 TFLOPS（白皮书 71，SM 时钟 1935 MHz），INT8 211～235 TOPS（白皮书 284，受 350 W 功耗墙限制），稀疏等效 252 TOPS，读带宽 887 GB/s（白皮书 936）。
  - Decode 受显存带宽限制，GPU 忙碌占周期 99% 以上。离带宽下限：bf16 1.19～1.21 倍，INT8 1.36～1.39 倍，稀疏 1.71～1.77 倍。差距的大头是线性层带宽利用率（bf16 81%～94%，INT8 70%～87%，稀疏 47%～77%）。
  - Prefill：D、F 和 bf16 的 A 受算力限制；INT8、稀疏的 A 受 CPU 发射限制。
  - `lm_head` 在所有路线都是 bf16，每步 1.40～1.43 ms，占 INT8、稀疏 TPOT 的 10%～13%。
- 验证：
  - 3 条路线 × A、D、F 各 3 次请求都完成，usage 断言通过。Decode 全部走 CUDA Graph。
  - 客户端 TPOT 和引擎周期在 A、F 中相差 ≤0.05 ms；Decode 前向 GPU 时间和 `bench forward` 相差 ≤2%；带宽下限的分项和与实际差相差 ≤0.03 ms。
  - 本地对 3 个 Python 文件做了 `ast.parse`，服务器上 import 成功，serve 正常运行。
  - 未验证：作用域的开销（只开 profiler 时生效）；`step()` 路径；前端作用域（前端 trace 为空，坑 64）。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-018：按 decoder 层拆分 kernel 时间

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/scripts/layer_breakdown.py`（新增）、`profiling/layer_breakdown.csv`、`profiling/report.md`（新增 2.3 节）、本文件。vLLM 源码未改动。
- 内容：复用 `forward/` 的 bench trace（本地副本），按层号和层内顺序拆分 bf16、INT8、稀疏的 d1、d8、t256、t8192 等 case。没有新的 GPU 测量。
- 原因：用户要求分析各层的性能。
- 结果：
  - 36 层耗时几乎相同，第 1～35 层的变异系数 ≤0.9%，没有慢层。
  - 第 0 层多 1%～7.5%，来自输入拷贝和补齐、未融合的输入 RMSNorm，以及第一个 qkv GEMM 慢 3～6 µs（推测是冷 L2，未验证）。
  - Decode 每层的 FFN 块占 55%～73%。Prefill t8192 时，INT8 和稀疏的 Attention 块占 52%～54%。
  - 每层离下限：Decode 1.20～1.85 倍，Prefill 1.15～1.60 倍。
- 验证：每个 case 的 36 层之和与第 2 节的整模型合计一致（如 bf16 d1 为 18.95 ms，t8192 为 1962.3 ms）。`ruff` 未运行。
- 状态：已提交并 push

### CHG-019：用自生成 GSM8K 数据校准 8B 的 OBR

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/scripts/gen_calib.py`（新增）、`scripts/obr_8b.py`（新增 `--calib`、`--root`、`--tag`）、`scripts/eval_route.py`（新增两条路线）、`eval/*selfgen*`、`eval/calib_selfgen.json`、`logs/selfgen.*`、`logs/obr_selfgen.log`、`report.md`、本文件。vLLM 源码未改动。
- 内容：
  - bf16 Qwen3-8B 用 GSM8K 评测的格式（`train[:5]` 作为 5 道例题，贪心，最多 256 token，遇 `Question` 停止），在 vLLM 里解答 `train[5:1205]`，平均 115 token。每个 2048 token 的窗口以 5 道例题（658 token）开头，后接生成的“题目 + 解答”，共 128 个窗口，用了 1078 对。生成的解答没有按对错筛选。
  - 其余设置和 CHG-016 相同（WANDA 掩码、PCG 60 次、逐块级联、INT8 和 INT4 共用一条链）。
  - 「不要重新踩的坑」新增第 72、73 条。
  - 这次由守卫脚本等 GPU 空闲后自动启动。profiling 停顿 8 分钟时被误判为结束，好在 profiling 当时已经完成，没有实际冲突。
- 原因：用户要求先回答“校准数据对不对路”，文献依据是 RESP（arXiv:2512.02185）和 SSGR（arXiv:2511.18864）。
- 结果：
  - INT8 + 2:4 + OBR：GSM8K 81.20%（1071/1319），WikiText 困惑度 21.24。WikiText 校准版是 44.43%、11.97。
  - INT4 + 2:4 + OBR：GSM8K 69.90%（922/1319），困惑度 28.29。WikiText 校准版是 19.56%、14.14。
  - 相对 bf16（88.55%）下降 7.35 个百分点，仍未达到 5 个百分点的标准，但差距从 44 个百分点缩小到 7 个。
  - 单层相对误差（INT8）：第 1 块 0.0026，中段最高约 0.018，末块 0.008。
- 验证：
  - 生成、OBR 和两条评测都跑完，rc=0。回答抽查是通顺、算式正确的解题过程。
  - 未验证：稀疏 kernel 在这个 checkpoint 上的精度（没有跑 `int8_obr_selfgen` + `cutlass_sparse`）；混合校准数据；其他任务上的泛化。这次只测了 1 次，没有重复。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-020：混合校准数据（WikiText + 自生成 GSM8K）

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/scripts/mix_calib.py`（新增）、`scripts/obr_8b.py`（新增 `--root4`）、`scripts/eval_route.py`、`eval/*_mix.*`、`logs/mix.*`、`logs/obr_mix.log`、`report.md`、本文件。vLLM 源码未改动。
- 内容：
  - 校准数据为 `obr_8b.py` 的 WikiText 窗口前 64 个（种子相同）加 `calib_selfgen.pt` 前 64 个，共 128 个。其余设置和 CHG-019 相同。
  - 为腾出磁盘，删除了服务器上的 `$W/ckpt/int8_24_nocomp`（可在 2 分钟内用 `quick_ckpt.py` 重新生成）。INT4 写到系统盘 `/root/obr_ckpt/int4_24_obr_mix`。
  - 「不要重新踩的坑」新增第 74 条。
- 原因：用户要求试混合校准，以兼顾 GSM8K 和通用困惑度。
- 结果：
  - INT8 + 2:4 + OBR：GSM8K 77.41%（1021/1319），困惑度 12.85。
  - INT4 + 2:4 + OBR：GSM8K 63.38%（836/1319），困惑度 15.14。
  - 对比纯 WikiText（44.43%、11.97）和纯自生成（81.20%、21.24）：困惑度几乎回到纯 WikiText 的水平，GSM8K 保留了纯自生成提升的 90%。
- 验证：
  - OBR 和两条评测都跑完，rc=0。只测了 1 次。
  - 未验证：混合比例的其他取值；稀疏 kernel 在这个 checkpoint 上的精度。`ruff`、`pre-commit`、`pytest` 未运行。
- 状态：已提交并 push

### CHG-019：bf16 压测：并发扫描、KV 满载、持续到达、分块对照和前端

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：`profiles/nsys/phase_cost/quant/obr_3090/scripts/stress_bf16.py`、`scripts/stress_summary.py`（新增）、`profiling/stress_summary.csv`、`profiling/report.md`（新增第 7 节）、本文件。vLLM 源码未改动。
- 内容：
  - 驱动等 GPU 空闲后自动开始（连续 3 次无计算进程），每个服务配置启动一次 `vllm serve`，用 `vllm bench serve` 跑各组合 3 次。同时以 0.5 s 间隔采集 `/metrics` 和 API 服务进程、EngineCore、客户端的 CPU，每 2 秒检查 GPU 上有没有外来进程。
  - 「不要重新踩的坑」新增第 72 至 76 条。
- 原因：用户要求压测：并发从 8 往上的吞吐、TPOT 和拐点，KV 用满后的抢占和排队，持续到达时 Prefill 对 Decode 的干扰，TTFT、TPOT 的 P95、P99 和 SLO，以及高并发流式下前端是否成为瓶颈。
- 结果（详见报告第 7 节）：
  - 吞吐拐点在并发 32（607 tok/s，并发 1 的 12.9 倍）。并发 ≥48 时 KV 100%，运行中封顶 44 个，3 次抢占 94～213 次，吞吐降到 541～553 tok/s，TPOT P99 超过 100 ms。
  - 长上下文（1536/384）只能同时运行 15 个，并发 32 时平均排队 14.8 s，TTFT P99 32 s。
  - Poisson 到达（1024/128）的服务能力约 2.2 req/s。ITL P95 在 2 req/s 起是 233～260 ms，等于一次 1024 token 的 Prefill。SLO-A（TTFT ≤1 s 且 TPOT ≤50 ms）达标率在 1、2、3 req/s 时分别为 100%、71.7%、3.3%。
  - `--max-num-batched-tokens 2048` 没有改善卡顿（见坑 76），3～4 req/s 的吞吐低 6.5%～8.5%，且 KV 少 13%。
  - 前端不是瓶颈：API 服务进程 CPU 平均 ≤10%、峰值 ≤28% 单核，因为 bf16 的流式输出最多只有约 600 tok/s。
- 验证：
  - 48 次压测全部 rc=0、failed=0，没有外来 GPU 进程。
  - c1 的引擎侧 ITL 20.4 ms 与客户端 20.3 ms 一致，也和第 3 节 profiling 的 TPOT 20.3 ms 一致。
  - 第一次启动因为坑 72、73 中止，已删除，没有混进结果。
  - 未验证：KV 满后吞吐下降中重算 Prefill 的占比；块 ≤512 的效果；`gpu-memory-utilization` 0.95；更高 token 速率下的前端。`ruff` 未运行。
- 状态：已提交并 push

### CHG-021：结束 OBR 实验，清理 3090 服务器

- 日期 / 作者：2026-11-07 / Claude Code
- 文件：本文件、`profiles/nsys/phase_cost/quant/obr_3090/report.md`。源码未改动。
- 内容：
  - 用户决定 OBR 2:4 稀疏实验到此为止，后续围绕 bf16 展开。
  - 删除了服务器上的全部量化 checkpoint：`$W/ckpt/` 下 4 个、`/root/obr_ckpt/` 下 3 个，共约 63GB。还删除了已备份到本地的 `eval`、`eval_signbug`、`res`、`prof` 和 `/tmp` 下的传输包。
  - 服务器保留（`W=/root/rivermind-data`）：`hf`（Qwen3-8B bf16）、`vllm`（venv 和打过补丁的代码，含 `_sparse_int8_sm80.so` 和 CHG-017 的作用域补丁）、`data`（WikiText2、GSM8K、`calib_*.pt`）、`obr/scripts`、`logs`、`roofline`、`stress`、`env.sh`。
  - 清理后系统盘剩 29GB，数据盘剩 44GB。
- 重新生成：量化 checkpoint 可以用 `scripts/quick_ckpt.py`（约 2 分钟）和 `scripts/obr_8b.py --calib ...`（约 22 分钟）重建，校准数据还在 `$W/data/`。
- 状态：已提交并 push

## 验证结果

以下为 CHG-004 之后的状态。除标注外，均在 RTX 5070、WSL2 上用 `--enforce-eager` 运行。

Step 序列与调度器逐步记录的对账：

| 场景 | Prefill | Decode | Mixed | 序列匹配 |
| ---- | ------- | ------ | ----- | -------- |
| `prefill_decode` | 1 | 7 | 0 | 8/8 |
| `chunked`（`max_num_batched_tokens=64`，含 1 token 尾块） | 8 | 6 | 0 | 14/14 |
| `mixed` | 1 | 36 | 3 | 40/40 |
| `prefix`（第二个请求从 `computed=112` 开始） | 2 | 6 | 0 | 8/8 |
| `preempt`（`max_model_len=160`，KV 24 MiB） | 5 | 121 | 1 | 127/127 |
| `mixed`，非 eager | 1 | 36 | 3 | 40/40，无 layer 标签 |
| `mixed`，`VLLM_NVTX_ENABLE=0` | 0 | 0 | 0 | 无本功能标签 |

eager 场景中，每个阶段的 layer 标签数都等于该阶段 Step 数，28 层齐全，且嵌套在同阶段 Step 内。除 `mixed` 外的 eager 场景都在 CHG-003 之后、CHG-004 之前运行；CHG-004 只影响非 eager，`mixed` 和非 eager 已在 CHG-004 之后重跑。

其他验证：

- `vllm serve` + `--capture-range=cudaProfilerApi` + `/start_profile`、`/stop_profile`：窗口内恰好 `Prefill 1 + Decode 4`，对应一次 5 token 请求。
- `vllm serve` + `nsys launch/start/stop`：结果相同。
- 两次 serve 日志都有 `EngineCore pid=...` 打出的安装信息，确认环境变量到达子进程。
- V1 runner（`VLLM_USE_V2_MODEL_RUNNER=0`）：打印 warning，运行完成。
- 吞吐（改动前 / 关闭 / 开启交替 4 轮）：795.9 / 830.5 / 839.9 tok/s。on/off 配对差异 -4.9% 至 +3.2%，测不出开销。
- 输出：开启和关闭插桩的运行全部与基线逐 token 相同；改动前的代码 5 次中 2 次不同（坑 5）。
- 15 个边界用例通过：标签格式、嵌套、异常时 push/pop 配对、关闭时不包裹、compiled 模型不被改动。

## 未验证与已知限制

- **未运行**：`ruff`、`pre-commit`、`pytest`。
- **只看过代码，未运行**：Qwen3 MoE。
- **未实测**：DBO、多 GPU 的 worker 子进程、Ray、投机解码的真实运行（`query_len>1` 的 Decode 只有单测）。
- **Mixed batch** 不拆分。同一层对整批只跑一次，拆不开。
- **DBO** 按整个 step 分类，微批不单独分类。
- **Elastic EP** 重入 warmup 时，已安装的 range 会给那次 warmup 打标签。
- **V1 runner** 没有实现本功能。
- **`pr-checklist`** 尚未运行。`AGENTS.md` 要求开 PR 前必须跑。

## 如何复现验证

验证脚本按用户要求不进仓库，放在会话临时目录，**可能已不存在**：

```text
/tmp/claude-1000/-home-bca-code-vllm/696e1ecf-28a9-4f6e-96a6-3e5d8d28244c/scratchpad/nvtx/
```

如果不存在，按下面的要点重建。

**调度器基准**：设 `VLLM_ENABLE_V1_MULTIPROCESSING=0`，让调度器在本进程内，然后从 `llm.llm_engine.engine_core.engine_core.scheduler` 取到它并包裹 `schedule()`。对每个 `total_num_scheduled_tokens > 0` 的输出：

- `computed = request.num_computed_tokens - num_scheduled`，因为 `schedule()` 返回时计数已经推进。
- `prefill_len` 取 `scheduled_new_reqs[i].prefill_token_ids` 的长度。V1 runner 下这个字段是 `None`。
- 预期阶段由 `computed < prefill_len` 得出。用 `llm_engine.add_request` 和 `llm_engine.step()` 逐步驱动，才能构造 Mixed。

**采集与导出**：

```bash
VLLM_NVTX_ENABLE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile -t cuda,nvtx,osrt --trace-fork-before-exec=true --force-overwrite true -o run python scenario.py
```

```bash
nsys export --type sqlite --force-overwrite true -o run.sqlite run.nsys-rep
```

**读取 NVTX 记录**（按 `start` 排序后，`Step/*` 序列应与基准逐项相同）：

```sql
SELECT COALESCE(e.text, s.value) AS label, e.start, e.end, e.globalTid
FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id
WHERE e.eventType = 59 AND e.end IS NOT NULL
ORDER BY e.start;
```

只比较标签数量不够：warmup 污染是在核对总数时发现的，定位则靠逐项比较序列。

**吞吐对照**：改动前的代码可以这样隔离，不动工作区：把 `vllm/` 复制到临时目录，用 `git show HEAD:<path>` 还原改动过的文件，删除新增文件，再用 `PYTHONPATH` 指向副本，并确认 `vllm.__file__`。
