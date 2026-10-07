/* SSD model mapping has a separate lifetime from its bounded expert cache. */
int ds4_gpu_set_stream_map(const void *model_map, uint64_t model_size,
        const uint64_t *offsets, const uint64_t *sizes, uint32_t count) {
    if (!model_map || !model_size || model_size > SIZE_MAX || !offsets || !sizes || !count) {
        return 0;
    }
    uint64_t end = 0u;
    for (uint32_t i = 0u; i < count; i++) {
        if (!sizes[i] || offsets[i] < end || offsets[i] > model_size ||
            sizes[i] > model_size - offsets[i]) { return 0; }
        end = offsets[i] + sizes[i];
    }
    const int src = cuda_mem_src_index(model_map);
    int device = 0;
    if (src >= DS4_MSRC_MAX || cudaGetDevice(&device) != cudaSuccess ||
        !cuda_model_coherent_host_access(device)) {
        fprintf(stderr, "ds4: SSD streaming requires coherent CUDA host page tables\n");
        return 0;
    }
    if (g_model_host_base == model_map && (g_model_registered || g_model_device_owned)) {
        return 0;
    }

    /* Full-map registration can pin cold routed weights. Keep the canonical
     * mmap pageable; the unit plan promotes only mandatory tensors, while
     * experts enter the separately charged bounded cache through pread. */
    cuda_model_preserve_current_direct_mapping();
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_registered = g_model_device_owned = 0;
    g_model_coherent_direct_map = model_map;
    g_model_hmm_direct = 1;
    g_model_range_mapping_supported = 0;
    g_model_cache_full = 0;
    g_model_needs_device_copy[src] = 0u;
    if (g_model_fd >= 0 && !g_model_fd_host_base) { g_model_fd_host_base = model_map; }
    return 1;
}
