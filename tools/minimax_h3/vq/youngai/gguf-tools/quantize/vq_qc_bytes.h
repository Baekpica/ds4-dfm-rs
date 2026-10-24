/* vq_qc_bytes.h — VQ 载荷字节账, 全仓唯一一份(2026-08-27 从 vq_qc.h 抽出)。
 *
 * 【为什么单独成头】体积分配器(rplan_solve)要在【量化前】算出每层会写多少字节, 量化器
 * (vq_qc.h)要在【写盘时】按同一公式排布。两处各抄一份 = 迟早对不上, 而对不上的表现是
 * "跑完 5 小时才发现超预算" 或 "分配器说 86G 实际写出 92G"。故只留这一份, 两边都 include。
 *
 * 布局(与 vq_fmt.h 解析端逐字对应): [16B 头][码本 nc*dim f16][行增益 rows f16][索引流]。
 * 索引流位宽 = ceil(log2 nc); 恰好 8 bit 时按字节存(无尾字节), 否则位流 +1 防解包尾读越界。
 */
#ifndef DS4_VQ_QC_BYTES_H
#define DS4_VQ_QC_BYTES_H
#include <stddef.h>

static size_t vq_idx_bytes(int rows,int cols,int dim,int nc){
    size_t nidx=(size_t)rows*cols/dim;
    int bits=1; while((1<<bits)<nc) bits++;
    if(bits==8) return nidx;                         /* 1B/索引(任意dim, q2正品无尾字节) */
    return (nidx*(size_t)bits+7)/8+1;                /* 位流 +1 防解包尾读越界(战役版语义) */
}
static size_t vq_payload_bytes(int rows,int cols,int dim,int nc){
    return 16 + (size_t)nc*dim*2 + (size_t)rows*2 + vq_idx_bytes(rows,cols,dim,nc);
}
#endif
