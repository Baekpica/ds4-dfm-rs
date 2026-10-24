/* deepseek4-quantize.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 2610 行实现体的纯字节
 * 切割(切点全在函数边界), 按序 cat p1..p6 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 文件头/CLI 约定注释 + die/xmalloc/路径/LE 读 + json 解析器
 * p2: safetensors 分片库(st_db) + fp8/fp4 反量化 + imatrix 加载
 * p3: hot_mask + 专家张量名解析 + HF↔GGUF 名映射表 + 量化策略(policy_type)
 * p4: f32→目标类型转换 + 专家/常规张量生成流水(线程池) + gguf 标量/字符串写
 * p5: gguf 元数据加载 + MTP 追加 + zchain 载入 + 输出上下文构建
 * p6: write_full_gguf 全量写出 + plan 打印 + usage/parse_args + main
 */
#include "../../src/common/ds4_amp_fmt.h"   /* z 秩上限等反修契约(单一定义源) */
#include "deepseek4-quantize_p1.inc.c"
#include "deepseek4-quantize_p2.inc.c"
#include "deepseek4-quantize_p3.inc.c"
#include "deepseek4-quantize_p4.inc.c"
#include "deepseek4-quantize_p5.inc.c"
#include "deepseek4-quantize_p6.inc.c"
