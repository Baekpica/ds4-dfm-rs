/* cuda/ds41_graph.cuh — decode-step CUDA graph primitives, ported from the C
 * engine at /data/YoungAi (commit 3946dbc), src/cuda/cuda_decode_graph.inc.cu.
 *
 * The engine captures a whole decode step (~1500 kernels) as one graph and
 * replays it with a single cudaGraphLaunch; the orchestration (when to
 * capture, bucket bounds, device positions, accounting) lives host-side in
 * ds4_ds41_graph.inc.  This file is only the primitive layer: begin/end
 * capture, launch, free, and the PDL edge rewrite.
 *
 * Port adaptations, named:
 *  - Stream.  The engine compiles -default-stream per-thread and captures on
 *    cudaStreamPerThread (every launch lands there).  The port launches on
 *    ds4_current_stream(), which is the legacy default stream (0) outside a
 *    capture -- and capture on stream 0 is forbidden by CUDA.  So the port
 *    captures on ONE dedicated stream created with default (blocking) flags
 *    and routes ds4_current_stream() onto it for the capture duration via the
 *    existing ds4_capture_set_stream.  The blocking flag is load-bearing, not
 *    cosmetic: the legacy default stream implicitly synchronizes with every
 *    blocking stream in BOTH directions (CUDA Programming Guide, implicit
 *    synchronization), so eager work on stream 0 -- notably v41_spec_rollback
 *    -- is ordered against a graph launched on this stream without extra
 *    events.  A nonblocking stream would NOT get that ordering and the next
 *    graph could read pre-rollback state.  Replay uses the same stream.
 *  - The routing must be restored on EVERY exit (failed begin, failed end,
 *    failed instantiate); a leaked routing would send later eager launches to
 *    the graph stream (the port's tensor-copy wrapper picks cudaMemcpyAsync on
 *    ds4_current_stream() while the routing is active).
 *  - Nested capture: the port's V4-era layer graphs share the same thread-local
 *    routing, so capture_begin refuses while any capture is active (the engine
 *    only ever had its own stream to check).
 *  - PDL edge rewrite is skipped below compute capability 9 (the engine only
 *    ever runs sm_121; on older devices the driver's error for a programmatic
 *    edge is not necessarily cudaErrorNotSupported, which the engine's
 *    per-edge fallback does not catch, and PDL does not exist there anyway).
 *    On sm_121 the rewrite and its fail-closed handling are the engine's.
 */
#pragma once
#include <stdint.h>
#include <stdio.h>

#include <cuda_runtime.h>

static void v41_pdl_register_small(void);   /* defined at the aggregation root ds4_ds41_gpu.cuh (the engine's ds4_cuda.cu:110-124) */

/* One blocking stream, created on first capture and kept for the process
 * (graphs replay on it; see the header comment for why it must be blocking). */
static cudaStream_t g_ds41_graph_stream = (cudaStream_t)0;

extern "C" int ds4_gpu_decode_graph_capture_begin(void) {
    v41_pdl_register_small();
    if (!g_ds41_graph_stream && cudaStreamCreate(&g_ds41_graph_stream) != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [graph] capture stream create failed\n");
        return 0;
    }
    if (ds4_capture_active()) return 0;   /* another capture (the V4-era layer graphs) owns the thread-local routing: never nest */
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(g_ds41_graph_stream, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        return 0;
    }
    /* ThreadLocal (the engine's choice): an unsafe call from THIS thread voids
     * the capture and capture_end reports NULL -- that is the safety net: the
     * caller falls back to direct dispatch instead of replaying a graph that
     * silently lost a step. */
    if (cudaStreamBeginCapture(g_ds41_graph_stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        fprintf(stderr, "ds4: [graph] capture begin failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    ds4_capture_set_stream(g_ds41_graph_stream);   /* route every ds4_current_stream() launch into the capture */
    return 1;
}

/* The engine's per-edge fallback discipline (cuda_decode_graph.inc.cu:33-74),
 * fail-closed: a library kernel whose params cannot be read keeps its ordinary
 * edge; cudaErrorNotSupported removing or adding an edge restores/keeps the
 * ordinary edge; any OTHER failure discards the whole graph -- a graph that
 * instantiates without a required dependency is worse than a loud direct
 * fallback. */
static uint64_t g_ds41_pdl_skipped = 0;
static int g_ds41_pdl_arch_ok = -1;   /* -1 untested / 0 skip the rewrite / 1 rewrite (device major >= 9) */
static int ds41_graph_pdl_edges(cudaGraph_t graph) {
    if (g_v41_pdl_n == 0) return 0;
    if (g_ds41_pdl_arch_ok < 0) {
        int major = 0;
        g_ds41_pdl_arch_ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess && major >= 9 ? 1 : 0;
        if (!g_ds41_pdl_arch_ok) {
            (void)cudaGetLastError();
            fprintf(stderr, "ds4: [graph] PDL edge rewrite skipped (device below compute capability 9)\n");
        }
    }
    if (!g_ds41_pdl_arch_ok) return 0;
    size_t ne = 0;
    if (cudaGraphGetEdges(graph, NULL, NULL, NULL, &ne) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    if (ne == 0) return 0;
    cudaGraphNode_t *from = (cudaGraphNode_t *)malloc(ne * sizeof *from), *to = (cudaGraphNode_t *)malloc(ne * sizeof *to);
    cudaGraphEdgeData *ed = (cudaGraphEdgeData *)malloc(ne * sizeof *ed);
    int n = 0, bad = 0, skipped = 0;
    if (!from || !to || !ed || cudaGraphGetEdges(graph, from, to, ed, &ne) != cudaSuccess) bad = 1;
    for (size_t i = 0; !bad && i < ne; i++) {
        if (ed[i].type != cudaGraphDependencyTypeDefault || ed[i].from_port != cudaGraphKernelNodePortDefault) continue;
        cudaGraphNodeType tf, tt;
        if (cudaGraphNodeGetType(from[i], &tf) != cudaSuccess || cudaGraphNodeGetType(to[i], &tt) != cudaSuccess) { bad = 1; break; }
        if (tf != cudaGraphNodeTypeKernel || tt != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp; memset(&kp, 0, sizeof kp);
        /* A node whose params cannot be read is a library kernel (cuBLAS): it
         * cannot be in the PDL registry, so its in-edges stay ordinary
         * (engine 2026-10-01: cuBLAS Sgemm nodes reported invalid device
         * function and dropped whole graphs until skipped here). */
        if (cudaGraphKernelNodeGetParams(to[i], &kp) != cudaSuccess) { (void)cudaGetLastError(); continue; }
        if (!v41_pdl_is_ready(kp.func)) continue;
        cudaGraphEdgeData pe; memset(&pe, 0, sizeof pe);
        pe.from_port = cudaGraphKernelNodePortLaunchCompletion; pe.type = cudaGraphDependencyTypeProgrammatic;
        cudaError_t er = cudaGraphRemoveDependencies(graph, &from[i], &to[i], &ed[i], 1);
        if (er == cudaErrorNotSupported) { (void)cudaGetLastError(); skipped++; continue; }
        if (er != cudaSuccess) { bad = 1; break; }
        const cudaError_t ea = cudaGraphAddDependencies(graph, &from[i], &to[i], &pe, 1);
        if (ea == cudaErrorNotSupported) {
            (void)cudaGetLastError();
            if (cudaGraphAddDependencies(graph, &from[i], &to[i], &ed[i], 1) != cudaSuccess) { bad = 1; break; }   /* put the original edge back */
            skipped++; continue;
        }
        if (ea != cudaSuccess) { bad = 1; break; }
        n++;
    }
    free(from); free(to); free(ed);
    if (bad) { fprintf(stderr, "ds4: [graph] PDL edge rewrite failed: %s\n", cudaGetErrorString(cudaGetLastError())); return -1; }
    if (skipped) g_ds41_pdl_skipped += skipped;
    return n;
}

/* Node-level dump for the decode-capture diff (DS41_GRAPH_NODE_DUMP=<prefix>):
 * one line per captured node -- sequence, kind, kernel name, launch geometry --
 * so the port's verify capture can be diffed against the engine's node by node
 * (docs/ds41-perf-2026-10-11.md, "The decode regime, resolved": 1788-1792
 * against 1460-1464).  Debug only: nothing is written unless the variable is
 * set, and captures are one-off per bucket. */
static void ds41_graph_node_dump(cudaGraph_t graph, const char *prefix) {
    static unsigned seq = 0;
    char path[512];
    snprintf(path, sizeof path, "%s.%u.txt", prefix, seq++);
    FILE *f = fopen(path, "w");
    if (!f) { return; }
    size_t nn = 0;
    if (cudaGraphGetNodes(graph, NULL, &nn) != cudaSuccess || nn == 0) { (void)cudaGetLastError(); fclose(f); return; }
    cudaGraphNode_t *nodes = (cudaGraphNode_t *)malloc(nn * sizeof *nodes);
    if (!nodes || cudaGraphGetNodes(graph, nodes, &nn) != cudaSuccess) { (void)cudaGetLastError(); free(nodes); fclose(f); return; }
    for (size_t i = 0; i < nn; i++) {
        cudaGraphNodeType t = cudaGraphNodeTypeEmpty;
        if (cudaGraphNodeGetType(nodes[i], &t) != cudaSuccess) { (void)cudaGetLastError(); }
        if (t == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams kp; memset(&kp, 0, sizeof kp);
            const char *nm = "?";
            if (cudaGraphKernelNodeGetParams(nodes[i], &kp) == cudaSuccess) { (void)cudaFuncGetName(&nm, kp.func); }
            else { (void)cudaGetLastError(); }
            fprintf(f, "%zu\tkernel\t%s\tg%ux%ux%u b%ux%ux%u s%zu\n", i, nm,
                    kp.gridDim.x, kp.gridDim.y, kp.gridDim.z,
                    kp.blockDim.x, kp.blockDim.y, kp.blockDim.z, kp.sharedMemBytes);
        } else if (t == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms mp; memset(&mp, 0, sizeof mp);
            unsigned long long bytes = 0;
            if (cudaGraphMemcpyNodeGetParams(nodes[i], &mp) == cudaSuccess) {
                bytes = (unsigned long long)mp.extent.width * (mp.extent.height ? mp.extent.height : 1u) *
                        (mp.extent.depth ? mp.extent.depth : 1u);
            } else { (void)cudaGetLastError(); }
            fprintf(f, "%zu\tmemcpy\t%llu\n", i, bytes);
        } else if (t == cudaGraphNodeTypeMemset) {
            fprintf(f, "%zu\tmemset\n", i);
        } else if (t == cudaGraphNodeTypeHost) {
            fprintf(f, "%zu\thost\n", i);
        } else if (t == cudaGraphNodeTypeGraph) {
            fprintf(f, "%zu\tsubgraph\n", i);
        } else {
            fprintf(f, "%zu\tother%d\n", i, (int)t);
        }
    }
    free(nodes);
    fclose(f);
}

extern "C" void *ds4_gpu_decode_graph_capture_end(void) {
    ds4_capture_set_stream((cudaStream_t)0);   /* restore on every exit: a leaked routing misdirects later eager launches */
    cudaGraph_t graph = NULL;
    const cudaError_t e = cudaStreamEndCapture(g_ds41_graph_stream, &graph);
    if (e != cudaSuccess || !graph) {
        fprintf(stderr, "ds4: [graph] capture voided: %s\n", cudaGetErrorString(e));
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        return NULL;
    }
    size_t n_nodes = 0;
    (void)cudaGraphGetNodes(graph, NULL, &n_nodes);
    const char *dump = getenv("DS41_GRAPH_NODE_DUMP");
    if (dump && *dump) { ds41_graph_node_dump(graph, dump); }
    const int n_pdl = ds41_graph_pdl_edges(graph);
    if (n_pdl < 0) { (void)cudaGraphDestroy(graph); return NULL; }
    cudaGraphExec_t exec = NULL;
    const cudaError_t ei = cudaGraphInstantiate(&exec, graph, 0);
    (void)cudaGraphDestroy(graph);
    if (ei != cudaSuccess || !exec) {
        fprintf(stderr, "ds4: [graph] instantiate failed (%zu nodes): %s\n", n_nodes, cudaGetErrorString(ei));
        (void)cudaGetLastError();
        return NULL;
    }
    fprintf(stderr, "ds4: [graph] decode step captured: %zu nodes (%d edges made programmatic, %llu unsupported kept ordinary)\n",
            n_nodes, n_pdl, (unsigned long long)g_ds41_pdl_skipped);
    return (void *)exec;
}

extern "C" int ds4_gpu_decode_graph_launch(void *exec) {
    if (!exec) return 0;
    return cuda_ok(cudaGraphLaunch((cudaGraphExec_t)exec, g_ds41_graph_stream), "decode graph launch");
}

extern "C" void ds4_gpu_decode_graph_free(void *exec) {
    if (exec) (void)cudaGraphExecDestroy((cudaGraphExec_t)exec);
}
