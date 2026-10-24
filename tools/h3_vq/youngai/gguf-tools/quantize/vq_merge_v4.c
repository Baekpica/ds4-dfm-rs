/* vq_merge_v4.c — 聚合根(2026-08-25 批6 拆分)。
 *
 * 单 TU 物理分片: 只按序 #include 下列 *.inc.c, 分片是原 947 行实现体的纯字节
 * 切割(切点在节注释边界), 按序 cat p1..p3 与拆分前逐字节一致。符号/数值零变化。
 *
 * p1: 基础工具 + 张量/条目 + GGUF 读写 + 命令行参数 + --extract-blobs/
 *     --extract-skeleton + 路由偏置(RBIA)
 * p2: ssh/本地流式取字节 + opt 内嵌(反修 op 编进合一 GGUF)
 * p3: --merge 合并写出 + main
 */
#include "vq_merge_v4_p1.inc.c"
#include "vq_merge_v4_p2.inc.c"
#include "vq_merge_v4_p3.inc.c"
