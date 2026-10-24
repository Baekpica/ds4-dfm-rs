/* v41_vq_rate.inc.cu — VQ 索引的"率侧"三件事(2026-09-21, 113 GB 方案探针), v41_vq.cu 单 TU include。
 *
 * ① 索引直方图 → 经验熵 H: 熵编码能省的比特 = log2(nc) − H, 这是"索引熵编码"这条路的全部本钱, 只能量不能猜
 *    (高斯源 8 维 Lloyd 码本的高分辨率理论值只有 0.13 bit, 重尾源会大得多 —— 权重是哪种, 直方图说了算)。
 * ② ECVQ 码长惩罚 pen[k] = λ·(−log2 p_k): 指派时 dist + pen, 同样的平均码长下换更低失真(Chou–Lookabaugh–Gray)。
 *    p_k 加半计数平滑: 没被选中的码字很贵但不是无穷, 下一轮质心挪过去还能被用上(纯 ECVQ 的"死码字"会永久出局, 探针不要这个耦合)。
 * ③ 嵌套码本分块定宽: 码字按频次排名, 每 B 个连续索引一块, 块宽 = 块内最大排名所需位宽 —— 这是"不破坏随机访问"的
 *    变长方案(块标签 + 定宽)能拿到的平均码长, 与 ① 的 H 一起画出"随机访问 vs 真熵编码"之间的差价。
 * 三件都只读索引, 不改任何产物字节; λ=0 且不要统计时一个核都不发。 */

__global__ static void vq_hist_kernel(const int *__restrict__ idx, long long nv, int *__restrict__ cnt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nv) atomicAdd(&cnt[idx[i]], 1);
}
__global__ static void vq_pen_kernel(const int *__restrict__ cnt, long long n, int nc, float lam, float *__restrict__ pen) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nc) return;
    const double p = ((double)cnt[k] + 0.5) / ((double)n + 0.5 * nc);
    pen[k] = (float)(-lam * log2(p));
}
/* 一线程一块: 块内最大排名 r ⇒ 位宽 bits(r+1)(r=0 也记 1 位: 全块同一码字的退化块不该算 0 字节) */
__global__ static void vq_blkw_kernel(const int *__restrict__ idx, const int *__restrict__ rank, long long nblk, int B,
                                      unsigned long long *__restrict__ wsum) {
    long long b = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    int mr = 0;
    for (int j = 0; j < B; j++) { const int r = rank[idx[b * B + j]]; if (r > mr) mr = r; }
    int w = 1; while ((1 << w) < mr + 1) w++;
    atomicAdd(wsum, (unsigned long long)w);
}
typedef struct { int cnt, k; } vq_ck;
static int vq_ck_cmp(const void *a, const void *b) {   /* 频次降序, 同频按码字号 —— 排序确定, 统计可复现 */
    const vq_ck *x = (const vq_ck *)a, *y = (const vq_ck *)b;
    return x->cnt != y->cnt ? (y->cnt > x->cnt) - (y->cnt < x->cnt) : (x->k > y->k) - (x->k < y->k);
}
/* 对【最终全量指派】的索引算率统计(idx 在 device, nv 个)。 */
static int vq_rate_stats(const int *idx, long long nv, int nc, v41_vq_rate *out) {
    int *cnt, *rank; unsigned long long *wsum;
    CK(cudaMalloc(&cnt, sizeof(int) * nc)); CK(cudaMalloc(&rank, sizeof(int) * nc)); CK(cudaMalloc(&wsum, sizeof(unsigned long long) * 2));
    CK(cudaMemset(cnt, 0, sizeof(int) * nc)); CK(cudaMemset(wsum, 0, sizeof(unsigned long long) * 2));
    vq_hist_kernel<<<(unsigned)((nv + 255) / 256), 256>>>(idx, nv, cnt);
    vq_ck *h = (vq_ck *)malloc(sizeof(vq_ck) * nc); int *hc = (int *)malloc(sizeof(int) * nc), *hr = (int *)malloc(sizeof(int) * nc);
    CK(cudaMemcpy(hc, cnt, sizeof(int) * nc, cudaMemcpyDeviceToHost));
    double H = 0, used = 0;
    for (int k = 0; k < nc; k++) { h[k].cnt = hc[k]; h[k].k = k; if (hc[k] > 0) { const double p = (double)hc[k] / nv; H -= p * log2(p); used += 1; } }
    qsort(h, (size_t)nc, sizeof(vq_ck), vq_ck_cmp);
    double half = 0;
    for (int r = 0; r < nc; r++) { hr[h[r].k] = r; if (r < nc / 2) half += h[r].cnt; }
    CK(cudaMemcpy(rank, hr, sizeof(int) * nc, cudaMemcpyHostToDevice));
    const long long nb16 = nv / 16, nb32 = nv / 32;   /* 行长 640/288 都是 32 的倍数 ⇒ 块不跨行 */
    vq_blkw_kernel<<<(unsigned)((nb16 + 255) / 256), 256>>>(idx, rank, nb16, 16, wsum);
    vq_blkw_kernel<<<(unsigned)((nb32 + 255) / 256), 256>>>(idx, rank, nb32, 32, wsum + 1);
    unsigned long long ws[2]; CK(cudaMemcpy(ws, wsum, sizeof ws, cudaMemcpyDeviceToHost));
    out->H = H; out->used = used; out->mass_half = half / nv;
    out->w16 = nb16 ? (double)ws[0] / nb16 : 0; out->w32 = nb32 ? (double)ws[1] / nb32 : 0;
    free(h); free(hc); free(hr); cudaFree(cnt); cudaFree(rank); cudaFree(wsum);
    return 0;
}
/* 码本舍到 E4M3 格点(仍存 f16 位型; E4M3 ⊂ f16, 后续解码路一个字不改)。为的是 nc8192 码本能以 8 B/词塞进 64 KB shared。 */
__global__ static void vq_round_e4m3_kernel(float *x, uint16_t *bits, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = ds4_e4m3fn_round(x[i]);
    x[i] = v; bits[i] = __half_as_ushort(__float2half_rn(v));
}
__global__ static void vq_f16_to_f32_kernel(const uint16_t *bits, float *x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__ushort_as_half(bits[i]));
}
