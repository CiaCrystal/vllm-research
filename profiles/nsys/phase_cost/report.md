# Qwen3-0.6B Prefill / Decode 开销分解（NVTX + nsys）

日期：2026-11-06。环境：RTX 5070，WSL2，`--enforce-eager`，Model Runner V2，异步调度开启。
所有数值是逐 step 值的中位数（Prefill 每轮 n=5，Decode 每轮 n=160），每配置独立进程跑 2 轮（r1、r2）。

**结果只代表 eager + CUPTI 追踪下的开销分布，不是生产配置的绝对耗时。** eager 没有 torch.compile、CUDA Graph 和 JIT warmup，Python 开销偏高。
`Step/*` 不含 logits 和采样，下表不计入它们。

## CPU range（ms；百分比是逐 step 占比的中位数）

| 运行 | 阶段 | Step | Attention | FFN | 层外 |
| ---- | ---- | ---- | --------- | --- | ---- |
| A_r1 | Prefill | 11.81 | 7.77 (65.6%) | 2.19 (19.1%) | 1.75 (14.7%) |
| A_r1 | Decode | 11.34 | 8.15 (71.8%) | 1.74 (15.2%) | 1.45 (12.8%) |
| A_r2 | Prefill | 12.49 | 8.27 (66.2%) | 2.36 (18.7%) | 1.85 (14.9%) |
| A_r2 | Decode | 11.59 | 8.38 (71.7%) | 1.76 (15.2%) | 1.49 (12.8%) |
| B_r1 | Prefill | 12.75 | 8.46 (67.3%) | 2.28 (18.3%) | 1.86 (14.5%) |
| B_r1 | Decode | 11.40 | 8.20 (72.3%) | 1.71 (15.0%) | 1.46 (12.8%) |
| B_r2 | Prefill | 11.17 | 7.41 (66.2%) | 2.09 (18.8%) | 1.69 (15.2%) |
| B_r2 | Decode | 11.44 | 8.28 (72.1%) | 1.71 (15.0%) | 1.46 (12.7%) |
| C_r1 | Prefill | 11.52 | 7.82 (67.5%) | 1.93 (17.6%) | 1.79 (15.5%) |
| C_r1 | Decode | 11.42 | 8.23 (72.1%) | 1.74 (15.3%) | 1.42 (12.6%) |
| C_r2 | Prefill | 11.67 | 7.83 (67.1%) | 2.09 (17.2%) | 1.70 (15.4%) |
| C_r2 | Decode | 11.46 | 8.27 (71.9%) | 1.78 (15.4%) | 1.44 (12.4%) |
| D_r1 | Prefill | 11.41 | 7.66 (68.2%) | 2.06 (17.2%) | 1.70 (14.9%) |
| D_r1 | Decode | 11.73 | 8.41 (71.9%) | 1.81 (15.4%) | 1.48 (12.7%) |
| D_r2 | Prefill | 11.99 | 7.88 (66.2%) | 2.06 (17.1%) | 1.84 (15.3%) |
| D_r2 | Decode | 11.73 | 8.47 (72.1%) | 1.80 (15.2%) | 1.47 (12.6%) |

## GPU 投影与 kernel 忙时（ms）

投影是首个 GPU 操作开始到最后一个结束的跨度，包含 kernel 之间的空闲。层外投影 = Step 投影 − 各层投影之和，因此也包含层与层之间的空闲，不等于层外 kernel 的开销。kernel 忙时才是 GPU 实际执行时间。

| 运行 | 阶段 | Step 投影 | Attention 投影 | FFN 投影 | 层外投影 | kernel 合计 | Attention | FFN | 层外 | 忙时/投影 |
| ---- | ---- | --------- | -------------- | -------- | -------- | ----------- | --------- | --- | ---- | --------- |
| A_r1 | Prefill | 11.74 | 7.46 | 2.48 | 1.71 | 4.67 | 2.40 (51%) | 2.18 (47%) | 0.10 (2.1%) | 40% |
| A_r1 | Decode | 11.30 | 7.60 | 1.53 | 2.21 | 2.01 | 0.99 (49%) | 0.94 (47%) | 0.07 (3.6%) | 18% |
| A_r2 | Prefill | 12.39 | 7.48 | 2.68 | 2.40 | 4.67 | 2.40 (51%) | 2.18 (47%) | 0.10 (2.1%) | 38% |
| A_r2 | Decode | 11.56 | 7.64 | 1.65 | 2.32 | 2.03 | 1.00 (49%) | 0.94 (46%) | 0.08 (3.8%) | 18% |
| B_r1 | Prefill | 19.90 | 10.76 | 8.34 | 0.80 | 18.80 | 10.22 (54%) | 8.19 (44%) | 0.28 (1.5%) | 94% |
| B_r1 | Decode | 11.29 | 7.60 | 1.58 | 2.22 | 2.26 | 1.23 (55%) | 0.94 (42%) | 0.08 (3.7%) | 20% |
| B_r2 | Prefill | 22.22 | 11.75 | 9.02 | 0.78 | 20.91 | 11.25 (54%) | 8.88 (42%) | 0.29 (1.4%) | 94% |
| B_r2 | Decode | 11.35 | 7.65 | 1.50 | 2.17 | 2.27 | 1.24 (55%) | 0.95 (42%) | 0.08 (3.7%) | 20% |
| C_r1 | Prefill | 33.77 | 16.26 | 16.50 | 1.15 | 32.81 | 15.74 (48%) | 16.39 (50%) | 0.69 (2.1%) | 97% |
| C_r1 | Decode | 11.25 | 7.71 | 1.66 | 1.93 | 2.92 | 1.72 (59%) | 1.11 (38%) | 0.09 (3.0%) | 26% |
| C_r2 | Prefill | 33.90 | 16.14 | 16.67 | 1.20 | 32.82 | 15.72 (48%) | 16.43 (50%) | 0.69 (2.1%) | 97% |
| C_r2 | Decode | 11.39 | 7.74 | 1.69 | 1.96 | 2.91 | 1.71 (59%) | 1.11 (38%) | 0.09 (3.0%) | 26% |
| D_r1 | Prefill | 151.33 | 79.02 | 68.94 | 3.46 | 150.39 | 78.66 (52%) | 68.81 (46%) | 2.96 (2.0%) | 99% |
| D_r1 | Decode | 11.58 | 8.26 | 1.65 | 1.64 | 4.04 | 2.83 (70%) | 1.12 (28%) | 0.09 (2.2%) | 35% |
| D_r2 | Prefill | 151.36 | 78.58 | 69.00 | 3.76 | 150.27 | 78.22 (52%) | 68.89 (46%) | 2.99 (2.0%) | 99% |
| D_r2 | Decode | 11.57 | 8.35 | 1.61 | 1.68 | 4.04 | 2.83 (70%) | 1.12 (28%) | 0.09 (2.2%) | 35% |

## Attention 内部 kernel（ms/step，括号为占 Attention kernel 时间的比例）

| 运行 | 阶段 | 合计 | 线性 | 注意力计算 | q/k RMSNorm | rotary | KV 写入 | 张量拷贝 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A_r1 | Prefill | 2.40 | 1.550 (64.5%) | 0.424 (17.6%) | 0.235 (9.8%) | 0.098 (4.1%) | 0.089 (3.7%) | 0.000 (0.0%) |
| A_r1 | Decode | 1.01 | 0.627 (62.3%) | 0.225 (22.3%) | 0.069 (6.9%) | 0.037 (3.7%) | 0.034 (3.4%) | 0.000 (0.0%) |
| A_r2 | Prefill | 2.40 | 1.551 (64.5%) | 0.420 (17.5%) | 0.235 (9.8%) | 0.099 (4.1%) | 0.090 (3.8%) | 0.000 (0.0%) |
| A_r2 | Decode | 1.01 | 0.631 (62.2%) | 0.228 (22.4%) | 0.071 (7.0%) | 0.036 (3.6%) | 0.034 (3.3%) | 0.000 (0.0%) |
| B_r1 | Prefill | 10.22 | 5.609 (54.9%) | 3.104 (30.4%) | 0.710 (6.9%) | 0.355 (3.5%) | 0.356 (3.5%) | 0.000 (0.0%) |
| B_r1 | Decode | 1.25 | 0.630 (50.5%) | 0.459 (36.8%) | 0.075 (6.0%) | 0.036 (2.9%) | 0.034 (2.7%) | 0.000 (0.0%) |
| B_r2 | Prefill | 11.25 | 6.439 (57.2%) | 3.308 (29.4%) | 0.710 (6.3%) | 0.354 (3.1%) | 0.355 (3.2%) | 0.000 (0.0%) |
| B_r2 | Decode | 1.25 | 0.635 (50.6%) | 0.459 (36.6%) | 0.075 (6.0%) | 0.036 (2.9%) | 0.034 (2.7%) | 0.000 (0.0%) |
| C_r1 | Prefill | 15.74 | 10.840 (68.9%) | 2.252 (14.3%) | 1.326 (8.4%) | 0.611 (3.9%) | 0.711 (4.5%) | 0.000 (0.0%) |
| C_r1 | Decode | 1.72 | 0.747 (43.5%) | 0.742 (43.2%) | 0.079 (4.6%) | 0.037 (2.1%) | 0.034 (2.0%) | 0.078 (4.5%) |
| C_r2 | Prefill | 15.72 | 10.834 (68.9%) | 2.249 (14.3%) | 1.325 (8.4%) | 0.610 (3.9%) | 0.706 (4.5%) | 0.000 (0.0%) |
| C_r2 | Decode | 1.71 | 0.747 (43.5%) | 0.742 (43.2%) | 0.079 (4.6%) | 0.037 (2.1%) | 0.034 (2.0%) | 0.078 (4.5%) |
| D_r1 | Prefill | 78.66 | 42.511 (54.0%) | 21.757 (27.7%) | 6.230 (7.9%) | 4.442 (5.6%) | 3.501 (4.5%) | 0.000 (0.0%) |
| D_r1 | Decode | 2.83 | 0.747 (26.4%) | 1.828 (64.7%) | 0.081 (2.9%) | 0.047 (1.7%) | 0.042 (1.5%) | 0.082 (2.9%) |
| D_r2 | Prefill | 78.22 | 42.183 (53.9%) | 21.552 (27.6%) | 6.236 (8.0%) | 4.450 (5.7%) | 3.493 (4.5%) | 0.000 (0.0%) |
| D_r2 | Decode | 2.83 | 0.747 (26.4%) | 1.829 (64.7%) | 0.081 (2.9%) | 0.047 (1.7%) | 0.042 (1.5%) | 0.082 (2.9%) |

## 每阶段前 5 kernel（ms/step，占该阶段 Step 内 kernel 时间）

| 配置 | 阶段 | 组 | kernel | r1 | r2 | 占比(r1) | 每 step 次数 |
| ---- | ---- | -- | ------ | -- | -- | -------- | ----------- |
| A | Prefill | FFN | `Kernel2` | 2.090 | 2.089 | 44.7% | 56 |
| A | Prefill | Attention | `Kernel2` | 1.550 | 1.551 | 33.2% | 56 |
| A | Prefill | Attention | `flash_fwd_splitkv_kernel` | 0.424 | 0.420 | 9.1% | 28 |
| A | Prefill | Attention | `rms_norm_kernel` | 0.235 | 0.235 | 5.0% | 56 |
| A | Prefill | Attention | `rotary_embedding_kernel` | 0.098 | 0.099 | 2.1% | 28 |
| A | Decode | FFN | `kernel` | 0.911 | 0.914 | 45.4% | 56 |
| A | Decode | Attention | `kernel` | 0.627 | 0.631 | 31.3% | 56 |
| A | Decode | Attention | `flash_fwd_splitkv_kernel` | 0.155 | 0.158 | 7.7% | 28 |
| A | Decode | Outer | `fused_add_rms_norm_kernel` | 0.070 | 0.073 | 3.5% | 56 |
| A | Decode | Attention | `flash_fwd_splitkv_combine_kernel` | 0.069 | 0.069 | 3.5% | 28 |
| B | Prefill | FFN | `Kernel2` | 7.975 | 8.658 | 42.4% | 56 |
| B | Prefill | Attention | `Kernel2` | 5.609 | 6.439 | 29.8% | 56 |
| B | Prefill | Attention | `flash_fwd_splitkv_kernel` | 3.104 | 3.308 | 16.5% | 28 |
| B | Prefill | Attention | `rms_norm_kernel` | 0.710 | 0.710 | 3.8% | 56 |
| B | Prefill | Attention | `reshape_and_cache_flash_kernel` | 0.356 | 0.355 | 1.9% | 28 |
| B | Decode | FFN | `kernel` | 0.910 | 0.917 | 40.3% | 56 |
| B | Decode | Attention | `kernel` | 0.630 | 0.635 | 27.9% | 56 |
| B | Decode | Attention | `flash_fwd_splitkv_kernel` | 0.365 | 0.365 | 16.2% | 28 |
| B | Decode | Attention | `flash_fwd_splitkv_combine_kernel` | 0.094 | 0.094 | 4.2% | 28 |
| B | Decode | Outer | `fused_add_rms_norm_kernel` | 0.080 | 0.080 | 3.6% | 56 |
| C | Prefill | FFN | `Kernel2` | 15.985 | 16.023 | 48.7% | 56 |
| C | Prefill | Attention | `Kernel2` | 10.840 | 10.834 | 33.0% | 56 |
| C | Prefill | Attention | `flash_fwd_splitkv_kernel` | 2.252 | 2.249 | 6.9% | 28 |
| C | Prefill | Attention | `rms_norm_kernel` | 1.326 | 1.325 | 4.0% | 56 |
| C | Prefill | Attention | `reshape_and_cache_flash_kernel` | 0.711 | 0.706 | 2.2% | 28 |
| C | Decode | FFN | `Kernel2` | 1.085 | 1.084 | 37.2% | 56 |
| C | Decode | Attention | `Kernel2` | 0.747 | 0.747 | 25.6% | 56 |
| C | Decode | Attention | `flash_fwd_splitkv_kernel` | 0.672 | 0.672 | 23.0% | 28 |
| C | Decode | Outer | `fused_add_rms_norm_kernel` | 0.080 | 0.080 | 2.7% | 56 |
| C | Decode | Attention | `rms_norm_kernel` | 0.079 | 0.079 | 2.7% | 56 |
| D | Prefill | FFN | `Kernel2` | 62.593 | 62.819 | 41.6% | 56 |
| D | Prefill | Attention | `Kernel2` | 42.511 | 42.183 | 28.3% | 56 |
| D | Prefill | Attention | `flash_fwd_splitkv_kernel` | 21.757 | 21.552 | 14.5% | 28 |
| D | Prefill | Attention | `rms_norm_kernel` | 6.230 | 6.236 | 4.1% | 56 |
| D | Prefill | FFN | `act_and_mul_kernel` | 6.049 | 6.065 | 4.0% | 28 |
| D | Decode | Attention | `flash_fwd_splitkv_kernel` | 1.754 | 1.756 | 43.4% | 28 |
| D | Decode | FFN | `Kernel2` | 1.094 | 1.095 | 27.1% | 56 |
| D | Decode | Attention | `Kernel2` | 0.747 | 0.747 | 18.5% | 56 |
| D | Decode | Outer | `fused_add_rms_norm_kernel` | 0.082 | 0.081 | 2.0% | 56 |
| D | Decode | Attention | `elementwise_kernel` | 0.082 | 0.082 | 2.0% | 56 |

## kernel 名称对照

由 sqlite 的 `demangledName` 确认：

- `Kernel2`：`cutlass::Kernel2<...gemm_bf16...>`，CUTLASS 矩阵乘。
- `kernel`：`internal::gemvx::kernel<...>`，cuBLAS 矩阵向量乘。批大小 1 的 Decode 走这条路径。
- `elementwise_kernel`：`at::native::...direct_copy_kernel_cuda`，torch 张量拷贝。只出现在批大小 8 的 Decode，每层 2 次，源码位置未定位。

线性类 kernel 归属（Attention 每层 2 次对应 `qkv_proj`、`o_proj`，FFN 每层 2 次对应 `gate_up_proj`、`down_proj`）是根据每层次数和 `Qwen3Attention.forward` 推断的，kernel 名称本身不含层名。

## Step 之外

数据来源：`nvtx_kern_sum:base` 中范围为空的行（kernel），以及 sqlite 里相邻 `Step/*` 的时间差（CPU 间隙）。两者都不计入上面任何占比。

- 范围外每 step 有 1 个线性类 kernel，中位数 0.47～0.48 ms，四个配置基本不变，按位置推断是 lm_head。采样相关 kernel 单个中位数不超过 0.004 ms。
- 相邻 Step 的 CPU 间隙中位数 1.3～1.5 ms。
- 例外是第一个 Decode Step 之后：C 为 10.0 ms，D 为 128.6 ms（r1）。异步调度下 CPU 先发射下一步，Prefill 的 GPU 积压在这里才被等待。这段等待不属于任何 Step。
- 单次重复从 Prefill 开始到最后一个 Decode 结束的跨度（r1/r2，ms）：A 435/471，B 458/447，C 470/459，D 583/591。

## 逐层结论

- GPU kernel 时间：Decode 中各层相对全层中位数的偏差不超过 2.4%，没有固定的离群层。
- Prefill 的 kernel 尖峰（A、B、D 中为 +7%～+229%）每个 step 落在不同的层，不是某层的固定开销。B_r2 的尖峰有 36 处（Attention、FFN 各 18），B_r1 为 5 处。B_r2 的 Prefill kernel 中位数因此比 B_r1 高约 11%，原因未确认。
- CPU 时间：`Layer_00` 是唯一稳定的离群层。Attention 在 Decode 中 8 轮全部高出 26%～35%，Prefill 合并 40 步高出 147%；FFN 在 Decode 中高出 7%～16%，Prefill 高出 55%。这只存在于 CPU 侧，GPU kernel 时间没有这个差异。
- `Layer_02`、`Layer_03` 的 Attention 在 Decode 中 8 轮都高出 6%～17%，幅度接近噪声，只记录不下结论。

## 实际运行的配置与校验

命令（`<name>`、`<prefix>` 按配置替换）：

    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers/nv_dispi.inf_amd64_20f49c6eb5c9b39b:$LD_LIBRARY_PATH
    export HF_HUB_OFFLINE=1 VLLM_NVTX_ENABLE=1 VLLM_SERVER_DEV_MODE=1
    nsys profile -t cuda,nvtx,osrt --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true -o <prefix> /home/bca/code/vllm/.venv/bin/vllm serve Qwen/Qwen3-0.6B --enforce-eager --port 8021 --gpu-memory-utilization 0.5 --no-enable-prefix-caching --profiler-config.profiler cuda --host 127.0.0.1 --max-model-len 2048 --max-num-batched-tokens 8192
    nsys stats -r nvtx_pushpop_sum,nvtx_gpu_proj_sum,nvtx_kern_sum:base --format csv --force-export=false --force-overwrite=true -o stats/<name> <name>.nsys-rep

每次请求：`POST /pause?mode=keep&clear_cache=false`，发出 completions，等 0.3 s，`POST /resume`。预热 4 次，形状与正式请求相同，在窗口外。

与任务书的偏离：

| 偏离 | 原因 | 证据 |
| ---- | ---- | ---- |
| `--max-model-len 2048` | 默认 40960 需要 4.38 GiB KV，0.5 只给 4.04 GiB，引擎启动失败 | `rejected/C_default.serve.log` |
| `--max-num-batched-tokens 8192` | 默认 2048，D 被切块 | `rejected/D_mml`：序列 `PMMMMD…` |
| `/pause` + `/resume`，需 `VLLM_SERVER_DEV_MODE=1` | 8 条 prompt 逐条到达，产生 Mixed | `rejected/C_mml`：序列 `PMD…` |
| `--host 127.0.0.1` | 开发接口不对外暴露 | 无 |
| 每配置 2 轮 | 实测跨进程噪声，任务书要求 1 轮 | 无 |

校验（8 个报告全部通过）：

- `Step/Prefill`=5，`Step/Decode`=160，无 `Step/Mixed`。序列逐项为 `P` + 32 个 `D`，重复 5 次。
- Prefill、Decode 各 28 层，每层 Attention、FFN 标签数都等于该阶段 Step 数。
- 日志都有 `added NVTX ranges to 56 modules`。
- `execute_context` 标签独立确认：Prefill 分别为 1(256)、1(1024)、8(2048)、8(8192)，Decode 为 1 或 8 条请求。
- 逐 step 还原与 `nsys stats` 对账：Step 的 CPU 中位数、投影中位数，单层投影中位数，以及 A_r1、D_r1 各范围的 kernel 总时间全部一致。

## 未确认的部分

- CUPTI 追踪本身对 CPU 时间的影响没有测，没有不带 nsys 的对照。
- Decode 有约 9%～14% 的 step 超过中位数 1.3 倍，最大约 45 ms。超出部分 94%～101% 在层内，单个 range 的占比中位数不超过 12%，在单次生成中的位置没有规律。原因未确认。
- Prefill 每轮只有 5 个样本，CPU 中位数两轮差异可达 13%（B）。
- 门控在窗口内引入每次 0.3 s 的空闲，影响未单独量化。
