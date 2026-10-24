/* v41_vq_pool.inc.cu — 共享码本探针的两个入口(2026-09-21), v41_vq.cu 单 TU include(须在 vq_prepare / vq_fit_assign 之后)。
 *
 * 【共享码本在量什么】09-21 上午实测 384 个专家的相对量化误差一致到 1.9% ⇒ 权重分布形状同构; 若一层只养一本码本,
 * 每专家的 64 KB 码本(GGUF 里三份 = 3.02 GB)全省掉。亏多少质量 = 池上训的一本 vs 每专家各训一本的残差差, 只能量。
 * 【怎么取样】从每个专家【归一化后】的向量流(与编码路同一个 vq_prepare, round_gain=1 ⇒ 池里的向量就是编码时看到的向量)
 * 按 stride 等距取样攒成一层的池, 池上 Lloyd 训一本; 之后每个专家拿它只指派不训练(v41_vq_opt.cb_fixed)。
 * 取样确定(等距, 无随机源), 池按专家号顺序拼接 ⇒ init_kernel 的等距初值天然跨专家。 */
static v41_dbuf g_pool_scratch = {NULL, 0}, g_pool_dw[8] = {{NULL, 0}};
/* 出厂 FP4 三矩阵上设备解成 f32(编码入口与取样入口共用同一组常驻缓冲) */
static int vq_upload_fp4(const uint8_t *const *w_host, const uint8_t *const *s_host, const int *rows, const int *cols, int nmat, float **w) {
    for (int m = 0; m < nmat; m++) {
        if (v41_dbuf_need(&g_pool_dw[m], sizeof(float) * (size_t)rows[m] * cols[m])) return -1;
        w[m] = (float *)g_pool_dw[m].p;
        if (v41_upload_dequant(w_host[m], s_host[m], "I8", rows[m], cols[m], 1, 32, w[m], &g_pool_scratch, NULL)) return -1;
    }
    return 0;
}
/* 一个专家的归一化向量按 stride 取样, 追加到主机池 out_host(容量 cap 个向量, *n_io 是已有个数, 回填后累加)。 */
extern "C" int v41_vq_pool_sample_from_fp4(const uint8_t *const *w_host, const uint8_t *const *s_host, const int *rows, const int *cols,
                                           int nmat, int dim, int stride, float *out_host, long long cap, long long *n_io) {
    float *w[8], *g[8], *V; uint16_t *gb[8]; long long nv;
    if (stride < 1 || dim <= 0 || dim > VQ_MAXDIM) { fprintf(stderr, "★取样参数不合法★\n"); return -1; }
    if (vq_upload_fp4(w_host, s_host, rows, cols, nmat, w)) return -1;
    for (int m = 0; m < nmat; m++) CK(cudaMalloc(&gb[m], sizeof(uint16_t) * rows[m]));
    if (vq_prepare(w, rows, cols, nmat, dim, 1, gb, g, &V, &nv)) return -1;
    const long long ntr = nv / stride;
    if (*n_io + ntr > cap) { fprintf(stderr, "★共享码本取样池满(%lld + %lld > %lld)★\n", *n_io, ntr, cap); return -1; }
    float *Vtr; CK(cudaMalloc(&Vtr, sizeof(float) * ntr * dim));
    gather_stride_kernel<<<(unsigned)((ntr + 255) / 256), 256>>>(V, Vtr, (int)ntr, dim, stride);
    CK(cudaMemcpy(out_host + *n_io * dim, Vtr, sizeof(float) * ntr * dim, cudaMemcpyDeviceToHost));
    *n_io += ntr;
    cudaFree(Vtr); cudaFree(V);
    for (int m = 0; m < nmat; m++) { cudaFree(g[m]); cudaFree(gb[m]); }
    return 0;
}
/* 池上训一本码本(Lloyd, 可带 ECVQ 惩罚 / E4M3 舍入), 回填 f16 位型 [nc×dim]。 */
extern "C" int v41_vq_train_codebook(const float *V_host, long long nv, int dim, int nc, int iters, float lam, int cb_fp8, uint16_t *cb_out) {
    float *V, *C; int *idx; uint16_t *cbb;
    CK(cudaMalloc(&V, sizeof(float) * nv * dim)); CK(cudaMemcpy(V, V_host, sizeof(float) * nv * dim, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&C, sizeof(float) * nc * dim)); CK(cudaMalloc(&cbb, sizeof(uint16_t) * nc * dim)); CK(cudaMalloc(&idx, sizeof(int) * nv));
    const vq_cw none = {NULL, 1, 0, NULL, 1};
    if (vq_fit_assign(V, nv, dim, nc, iters, 1, C, idx, none, 1, cbb, lam, 0, cb_fp8, NULL)) return -1;
    CK(cudaMemcpy(cb_out, cbb, sizeof(uint16_t) * nc * dim, cudaMemcpyDeviceToHost));
    cudaFree(V); cudaFree(C); cudaFree(cbb); cudaFree(idx);
    return 0;
}
