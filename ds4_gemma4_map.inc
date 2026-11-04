/* Gemma-4 discrete SSD stream map (RTX 4070 SUPER / sm_89).
 *
 * The shared map, ds4_glm53_map.inc:2-38 (GLM53-owned, never edited here),
 * refuses without coherent CUDA host page tables. That gate is exactly right
 * for the shape it was written for: on GB10 the mandatory resident spans stay
 * in the model mmap and kernels read them through ATS
 * (g_model_hmm_direct = 1, g_model_range_mapping_supported = 0), while the
 * routed experts stay pageable until the slot pool prereads them. A discrete
 * card has no pageable memory access (cudaDevAttrPageableMemoryAccess = 0 and
 * cudaDevAttrPageableMemoryAccessUsesHostPageTables = 0, measured on the 4070
 * SUPER), so a host pointer is not device readable and the resident spans must
 * become device-resident instead of HMM-direct.
 *
 * This file is that discrete arm, built out of the engine's own parts:
 *
 *  1. ds4_gpu_set_stream_map_gemma installs the same stream-map state the
 *     shared function installs (canonical mmap left UNregistered, experts
 *     host-cold, fd owner bound, per-range registration off) minus the
 *     coherence gate, and then promotes every mandatory resident span through
 *     the eager materializer's own unit copy (cuda_unit_materialize_copy:
 *     O_DIRECT / shard-fd read, source-page discard, range publication). The
 *     spans are the ones gemma4_stream_spans (ds4_gemma4_stream.inc:276)
 *     already hands the map call: every tensor except the routed expert VQ
 *     and residual records. After this, cuda_model_range_ptr serves each
 *     resident tensor from the registry (device bytes), never from the mmap.
 *  2. ds4_gpu_g4_stream_plan_cold publishes ONE 128-byte device range per
 *     cold expert record, holding that record's real GVQ1/GVR1 header. The
 *     binder (ds4_glmvq_gpu.cuh:235-261 gvq_headers) gathers every VQ/RES
 *     record's header with a device kernel, so the record start must be
 *     device readable even for an expert that is never resident. The range
 *     covers exactly GVQ_HEADER bytes: a later whole-record request cannot be
 *     served a truncated pointer, because containment fails and the resolve
 *     misses. gemma4_gpu_bind marks those records header-sized and plain for
 *     the bind, then ds4_gpu_g4_stream_plan_restore puts the truthful record
 *     back; ds4_gpu_g4_stream_rebind re-validates every one of them from the
 *     pread staging before the first routed read (ds4_gemma4_gpu.cuh:974).
 *
 * Nothing here runs unless --ssd-streaming installed a Gemma stream map, the
 * card lacks coherent host page tables, and the family is Gemma-4. The
 * resident (default) path never reaches this file, and the GB10 arm is the
 * shared function unchanged. */

#ifndef DS4_GEMMA4_MAP_INC
#define DS4_GEMMA4_MAP_INC

/* Set when the discrete arm installed the stream map; read by the Gemma
 * binder to choose the header-sized cold records. */
static int g_g4_stream_discrete_plan;
/* One device buffer holding every cold record's GVQ_HEADER bytes. The
 * published ranges point into it; it lives until the engine closes. */
static ds4_gpu_tensor *g_g4_stream_headers;

extern "C" int ds4_gpu_g4_stream_discrete_plan(void) {
    return g_g4_stream_discrete_plan;
}

extern "C" void ds4_gpu_g4_stream_plan_release(void) {
    ds4_gpu_tensor_free(g_g4_stream_headers);
    g_g4_stream_headers = NULL;
    g_g4_stream_discrete_plan = 0;
}

/* The 128-byte header of every cold expert record, plus one device range per
 * record. Called before the GVQ plan binds, from gemma4_gpu_bind. */
extern "C" int ds4_gpu_g4_stream_plan_cold(const void *model_map,
        uint64_t model_size, const uint64_t *offsets, const uint64_t *bytes,
        uint32_t count, const uint8_t *headers, uint64_t header_bytes) {
    if (!model_map || !model_size || model_size > SIZE_MAX) return 0;
    if (count == 0) return 1;
    if (!offsets || !bytes || !headers || !header_bytes || count > UINT32_MAX) {
        return 0;
    }
    if (g_g4_stream_headers) {
        fprintf(stderr, "ds4: Gemma-4 SSD cold header plan already installed\n");
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (bytes[i] != header_bytes || offsets[i] > model_size ||
            bytes[i] > model_size - offsets[i]) {
            fprintf(stderr, "ds4: Gemma-4 SSD cold header span %u is invalid\n", i);
            return 0;
        }
        /* Sorted, disjoint record starts are a property of the tensor table;
         * an overlap would mean the same record was listed twice and its
         * second publication would be refused below. */
        if (i && offsets[i] <= offsets[i - 1u]) {
            fprintf(stderr, "ds4: Gemma-4 SSD cold header offsets are not "
                            "strictly increasing at %u\n", i);
            return 0;
        }
    }
    const uint64_t total = (uint64_t)count * header_bytes;
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(total);
    if (!t) { return 0; }
    if (!ds4_gpu_tensor_write(t, 0, headers, total)) {
        ds4_gpu_tensor_free(t);
        return 0;
    }
    for (uint32_t i = 0; i < count; i++) {
        cuda_model_range r = {};
        r.host_base = model_map;
        r.offset = offsets[i];
        r.bytes = bytes[i];
        r.device_ptr = (char *)t->ptr + (uint64_t)i * header_bytes;
        if (!cuda_model_range_publish(r)) {
            fprintf(stderr, "ds4: Gemma-4 SSD cold header range %u at %llu "
                            "was refused\n", i, (unsigned long long)offsets[i]);
            /* Every range already published stays; the bind aborts, and the
             * engine unwinds the whole open. */
            return 0;
        }
    }
    g_model_range_bytes += total;
    g_g4_stream_headers = t;
    fprintf(stderr,
            "ds4: Gemma-4 SSD cold records: %u headers, %.2f MiB device "
            "(experts stay host-cold; each is rebound from its pread slot)\n",
            count, (double)total / 1048576.0);
    return 1;
}

/* Undo the bind-time stand-in: the expert records go back to their truthful
 * type, bytes and book/res links, and their source pointer is cleared. The
 * first ds4_gpu_g4_stream_rebind fills each one from the slot pool; until
 * then the entry is a NULL, not a host pointer a kernel could read. */
extern "C" int ds4_gpu_g4_stream_plan_restore(ds4_gvq_model *plan,
        const ds4_gvq_record *truth, const uint32_t *cold, uint32_t count) {
    if (!plan || !truth || (count && !cold)) return 0;
    const uint32_t n = (uint32_t)plan->records.size();
    if (n != plan->source.size()) return 0;
    for (uint32_t k = 0; k < count; k++) {
        const uint32_t i = cold[k];
        if (i >= n) return 0;
        plan->records[i] = truth[i];
        plan->source[i].data = nullptr;
        plan->source[i].scale = nullptr;
        plan->source[i].vq = glmvq_matrix{};
        plan->source[i].storage = GLMVQ_WEIGHT_BF16;
        plan->source[i].rows = truth[i].rows;
        plan->source[i].cols = truth[i].cols;
    }
    const uint64_t bytes = (uint64_t)n * sizeof(glmvq_source);
    if (!plan->device || !ds4_gpu_tensor_write(plan->device, 0,
                                              plan->source.data(), bytes)) {
        return 0;
    }
    return 1;
}

/* The discrete stream map. Coherent cards keep the shared implementation
 * byte for byte; this is the only branch that differs. */
extern "C" int ds4_gpu_set_stream_map_gemma(const void *model_map,
        uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes,
        uint32_t count) {
    if (!model_map || !model_size || model_size > SIZE_MAX || !offsets ||
        !sizes || !count) {
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
    if (src >= DS4_MSRC_MAX || cudaGetDevice(&device) != cudaSuccess) {
        fprintf(stderr, "ds4: SSD streaming requires coherent CUDA host page tables\n");
        return 0;
    }
    if (cuda_model_coherent_host_access(device)) {
        /* GB10: the shared contract, unchanged. */
        return ds4_gpu_set_stream_map(model_map, model_size, offsets, sizes, count);
    }
    if (cuda_model_source_host_mapped(model_map)) {
        fprintf(stderr, "ds4: Gemma-4 SSD streaming conflicts with a terminal "
                        "host-mapped weight residency\n");
        return 0;
    }

    /* Same state the shared function installs, minus the coherence facts: the
     * canonical mmap stays UNregistered (registering it would pin the whole
     * artifact, cold experts included), no ATS fact is recorded, and the
     * per-range register tier is off so a stray expert resolve can never pin
     * a routed record into host RAM. */
    cuda_model_preserve_current_direct_mapping();
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_registered = g_model_device_owned = 0;
    g_model_coherent_direct_map = NULL;
    g_model_hmm_direct = 0;
    g_model_range_mapping_supported = 0;
    g_model_cache_full = 0;
    g_model_needs_device_copy[src] = 0u;
    if (g_model_fd >= 0 && !g_model_fd_host_base) { g_model_fd_host_base = model_map; }
    g_g4_stream_discrete_plan = 1;

    /* Eager promotion of the mandatory resident spans through the unit
     * materializer (the eager pass's own copy: O_DIRECT when the map owns the
     * fd or the shards, source-page discard, funnel publication). */
    uint64_t populated = 0, promote = 0, failed = 0, unfunded = 0;
    for (uint32_t i = 0; i < count; i++) {
        ds4_phys_unit u = {};
        u.src_off = offsets[i];
        u.src_bytes = sizes[i];
        int exact_hit = 0;
        if (cuda_model_unit_device_covered(model_map, u.src_off, u.src_bytes,
                                           &exact_hit)) {
            continue;
        }
        promote++;
        const uint64_t limit = cuda_model_cache_limit_bytes();
        const int legacy = (g_model_range_bytes >= limit ||
                            u.src_bytes > limit - g_model_range_bytes)
                               ? DS4_GOV_REFUSE_CLASS : DS4_GOV_ADMIT;
        ds4_gov_claim mcl = {};
        mcl.requester = DS4_GOVC_ENGINE_BOOT;
        mcl.memc = DS4_MEMC_WEIGHT_ARENA;
        mcl.domain = DS4_MEMD_UNIFIED_DEVICE;
        mcl.proposed_outstanding = g_model_range_bytes + u.src_bytes;
        mcl.class_limit = limit ? limit : 1;
        if (ds4_gov_governed_check("gemma_stream_materialize", &mcl, legacy)
                != DS4_GOV_ADMIT) {
            unfunded++;
            continue;
        }
        char label[64];
        snprintf(label, sizeof(label), "gemma4-resident:%u", i);
        const int oc = cuda_unit_materialize_copy(model_map, &u,
                                                  /*direct_discard=*/1, label);
        if (oc == DS4_RESUNIT_POPULATED) {
            populated += u.src_bytes;
            ds4_gov_publish_use(DS4_GOVC_ENGINE_BOOT, g_model_range_bytes,
                                g_model_range_bytes);
        } else if (oc == DS4_RESUNIT_FAILED) {
            failed++;
        }
    }
    cuda_model_stage_pool_release();
    fprintf(stderr,
            "ds4: Gemma-4 SSD discrete map: %u resident spans, %u promoted, "
            "%.2f GiB device-resident, %llu unfunded, %llu failed "
            "(mmap unregistered, experts host-cold)\n",
            count, (uint32_t)promote, (double)populated / 1073741824.0,
            (unsigned long long)unfunded, (unsigned long long)failed);
    if (failed) {
        fprintf(stderr, "ds4: Gemma-4 SSD discrete map could not make the "
                        "resident spans device-resident\n");
        return 0;
    }
    if (unfunded) {
        fprintf(stderr, "ds4: Gemma-4 SSD discrete map: %llu resident spans did "
                        "not fit the weight cache budget\n",
                (unsigned long long)unfunded);
        return 0;
    }
    return 1;
}

#endif
