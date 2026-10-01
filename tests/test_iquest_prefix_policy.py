#!/usr/bin/env python3
"""Check prefix-probe owner admission and import ordering without a GPU."""
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / 'tests/test_iquest_prefix.c').read_text()
    assert 'ds4_gpu_import_model_ipc_manifest' in source, 'Prefix probe must import its weight owner'
    helpers = source.split('#include "../ds4.c"', 1)[1].split('static bool dump_stage', 1)[0]
    harness = r'''
#include "ds4.h"
#include "ds4_gpu.h"
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { void *map; uint64_t size; unsigned split_count; int fd; } ds4_model;
static int step, fail;
static int advance(int expected) { assert(++step == expected); return step != fail; }
int ds4_gpu_iquest_policy(void) { return advance(1); }
int ds4_gpu_init(void) { return advance(2); }
int ds4_gpu_set_model_fd(int fd) { assert(fd == -1); return advance(3); }
int ds4_gpu_model_source_bind(const void *p,uint64_t n,int role,int fd,int policy,const char *name,const char *path) {
    assert(p && n == 128 && role == DS4_MSRC_ROLE_PRIMARY && fd == -1);
    assert(policy == DS4_RESIDENCY_HOST_MAPPED && !strcmp(name,"base") && !strcmp(path,"model.gguf"));
    return advance(4) ? 0 : -1;
}
int ds4_gpu_set_model_map(const void *p,uint64_t n) { assert(p && n == 128); return advance(5); }
int ds4_gpu_import_model_ipc_manifest(const void *p,uint64_t n,const char *manifest,const char *id) {
    assert(p && n == 128 && !strcmp(manifest,"owner.manifest") && !strcmp(id,"base"));
    return advance(6);
}
static void model_release_mapping_cache(ds4_model *m) { assert(m); assert(advance(7)); }
HELPERS
int main(int argc,char **argv) {
    assert(argc == 2);
    if (!strcmp(argv[1],"env")) { return prefix_manifest() ? 0 : 1; }
    char bytes[128]; ds4_model m = {bytes,sizeof(bytes),6,99};
    fail = atoi(argv[1]);
    const bool ok = prefix_weights(&m,"model.gguf","owner.manifest");
    assert(ok == (fail == 0));
    assert(step == (fail ? fail : 7));
    return 0;
}
'''.replace('HELPERS', helpers)
    with tempfile.TemporaryDirectory(prefix='iquest-prefix-policy-') as work:
        code, binary = Path(work) / 'check.c', Path(work) / 'check'
        code.write_text(harness)
        subprocess.run(['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT),
                        str(code), '-o', str(binary)], check=True)
        for failed_step in range(7):
            subprocess.run([str(binary), str(failed_step)], check=True)
        clean = {k: v for k, v in os.environ.items() if not k.startswith('DS4_')}
        cases = [({}, False), ({'DS4_CUDA_WEIGHT_IPC_MANIFEST': ''}, False)]
        owner = {'DS4_CUDA_WEIGHT_IPC_MANIFEST': 'owner.manifest'}
        cases += [(owner, True)]
        for scope in ('base', 'both', '', 'mtp', 'invalid'):
            cases.append((owner | {'DS4_CUDA_WEIGHT_IPC_SCOPE': scope}, scope in ('base', 'both', '')))
        cases += [(owner | {'DS4_CUDA_COPY_MODEL': '1'}, False)]
        for env, accepted in cases:
            result = subprocess.run([str(binary), 'env'], env=clean | env, capture_output=True)
            assert (result.returncode == 0) == accepted, (env, result.stderr.decode())
    print('IQuest prefix owner policy: 7 lifecycle paths and 9 admission cases passed')


if __name__ == '__main__':
    main()
