/* quants.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 本文件只按序 #include 下列 *.inc.c, 分片是原 1206 行实现体的
 * 纯字节切割(切点全在函数边界), 按序 cat quants_p1..p3 与拆分前逐字节一致。
 * 符号/编译单元/数值零变化 —— 想看完整实现就按序读分片。
 *
 * p1: 头/公共辅助(qkx2/qkx3/qp) + q8_0 + q4_K 参考路
 * p2: q4_K 加权路 + q2_K 家族 + iq2_xxs 网格初始化/导出
 * p3: iq2_xxs 块编码/CPU 编码器/GPU 挂点 + 类型分发 + f16/bf16 行转换
 */
#include "quants_p1.inc.c"
#include "quants_p2.inc.c"
#include "quants_p3.inc.c"
