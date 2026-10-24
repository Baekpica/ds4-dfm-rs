/* ds4_amp_fmt.h — 反修产物跨域契约: 单一定义源(2026-08-31 魔数扫除)。
 *
 * 这些数字同时被 引擎(ds4_zchain.c / src/core/core_sidecar.c)、CUDA
 * (src/cuda/cuda_zchain_*.inc.cu)、量化工具(gguf-tools/amp, gguf-tools/quantize)
 * 消费。历史上各写一份字面量(λ clamp 8 份、116 记录头 12 份、3/4 切分 7 份),
 * 改任何一份都会让判决尺与部署路静默分叉 —— 从此只许改这里。
 *
 * ★metal 目录的 .metal shader 无法 include 本头(源码运行时拼接编译, 无宿主 include 路径):
 * metal/moe_sidecar.metal 的 clamp(0.25,4) 与 threadgroup pvS[1024] 是同一契约的
 * 镜像字面量, 改这里必须同步改那边(那边已留指回注释)。 */
#ifndef DS4_AMP_FMT_H
#define DS4_AMP_FMT_H

/* dql/zrec 落地修正记录: name16 + algo64 + vol8 + psz8 + mean16 + vd4 = 116 字节头, 载荷紧随 */
#define DS4_AMP_REC_HDR      116
#define DS4_AMP_REC_OFF_VOL   80   /* u64 载荷体积 */
#define DS4_AMP_REC_OFF_PSZ   88   /* u64 载荷字节数 */
#define DS4_AMP_REC_OFF_MEAN 108   /* f32 均值(诊断用, mean16 区的第 4 个 f32) */
#define DS4_AMP_REC_OFF_VD   112   /* i32 判决位(1=落地生效) */

/* λ 缩放链(GL/dyn2/dyn8)夹取区间: 引擎 CPU/CUDA/Metal 与工具回放必须逐位同。
 * 0.25 与 4.0 在 float/double 下均精确可表示, 各消费点按各自精度 cast 不引入偏差。 */
#define DS4_AMP_LAM_MIN 0.25
#define DS4_AMP_LAM_MAX 4.0

/* z^L 秩硬上限: metal/moe_sidecar.metal 的 threadgroup 定长数组 pvS[1024] 决定,
 * 超过它=GPU 端越界踩共享内存(08-30 band kernel 错位同族事故)。提秩必须先改 shader。 */
#define DS4_AMP_ZK_MAX 1024

/* z^L 信任域默认(‖Δ‖≤tr·‖routed‖ 的 tr): 载荷 tr 槽的生产写值。曾在 zlayer_p7 两处
 * 与 zloss_emit 各写一份 0.5f 字面量 —— 部署强度是判决尺/引擎/解算评估共用的契约,
 * 散写=改一处漏两处的静默分叉。tr≤0 无部署编码(引擎把 cap=0 读成整条修正清零)。 */
#define DS4_AMP_ZL_TR 0.5f

/* fit/val 切分: 前 3/4 拟合、尾 1/4 验证。ds4quant_run_p14 的 checkpoint 续跑用同一
 * 公式从旧 S 反推旧切分 —— 改比例=历史 checkpoint 全部静默错切, 视为格式冻结。 */
#define DS4_AMP_FIT_SPLIT(n) (((n) * 3) / 4)

/* z 族 op 载荷布局: 16B 头(u32 k | f32 tr/s | u32 din | u32 dout) + fp16 元素。
 * 元素数按形态分两种; 写者/读者共 10+ 处必须逐位同, 从此只认这两个公式。 */
#define DS4_AMP_OP_HDR 16
#define DS4_AMP_ZL_ELEMS(k, din, dout)   ((size_t)(k) + (size_t)(k) * (din) + (size_t)(k) * (dout))   /* z[k]|U[dout·k]|V[din·k] (type6/7/8) */
#define DS4_AMP_AMPD_ELEMS(k, din, dout) (2u * (size_t)(k) * (din) + (size_t)(k) * (dout))            /* A[din·k]|U|V (type9 zl.AMPD) */

/* 合一 GGUF opt_chain: 每 op 16 个 f32 槽。槽位: f[0]=type, f[1]=g/tr, f[2..5]=w2p
 * (type6: f[2]=k, f[3]=din, 0=D), f[6..14]=w8, f[15]=v8 块号(-1 无)。 */
#define DS4_AMP_CHAIN_FLOATS 16

#endif
