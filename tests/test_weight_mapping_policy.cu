// Exercise the actual common map API, not an IQuest-only allocation policy.
#include <cuda_runtime.h>
extern "C" {
#include "../ds4.h"
}
#include "../ds4_gpu.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char **argv) {
    if(argc<3 || argc>4)return 2;
    const bool chunked=argc==4;
    const size_t n=1u<<20;const off_t off=strtoull(argv[2],nullptr,10)&~4095ULL;
    int fd=open(argv[1],O_RDONLY);void *map=mmap(nullptr,n,PROT_READ | (chunked?PROT_WRITE:0),MAP_PRIVATE,fd,off);
    if(fd<0 || map==MAP_FAILED || !ds4_gpu_init())return 2;
    int device=0,ro=0,ats=0;cudaGetDevice(&device);
    cudaDeviceGetAttribute(&ro,cudaDevAttrHostRegisterReadOnlySupported,device);
    cudaDeviceGetAttribute(&ats,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device);
    if(!chunked && (ro || ats)) {fprintf(stderr,"Use a device without read-only mapping for rejection control\n");return 77;}
    ds4_gpu_model_source_bind(map,n,DS4_MSRC_ROLE_PRIMARY,fd,DS4_RESIDENCY_HOST_MAPPED,"probe",argv[1]);
    if(chunked){setenv("DS4_CUDA_COPY_MODEL_CHUNKED","1",1);setenv("DS4_CUDA_MODEL_COPY_CHUNK_MB","16",1);}
    const uint64_t offsets[1]={0},sizes[1]={n};
    const int accepted=chunked ? ds4_gpu_set_model_map_spans(map,n,offsets,sizes,1,n) : ds4_gpu_set_model_map(map,n);
    ds4_mem_cell span={},whole={};ds4_gpu_mem_census_read(DS4_MEMC_WEIGHT_SPAN,DS4_MEMD_UNIFIED_DEVICE,&span);
    ds4_gpu_mem_census_read(DS4_MEMC_WEIGHT_WHOLE,DS4_MEMD_UNIFIED_DEVICE,&whole);
    printf("{\"terminal_mapped_rejected\":%s,\"canonical_device_copy_bytes\":%llu,\"scope\":\"actual-native-map-api-real-one-MiB-file-region\",\"control\":\"%s\"}\n",accepted?"false":"true",(unsigned long long)(span.committed-span.freed_committed+whole.committed-whole.freed_committed),chunked?"bounded-writable-chunk-copy-conflict":"readonly-registration-unavailable");
    ds4_gpu_unregister_model_map(map);ds4_gpu_cleanup();munmap(map,n);close(fd);
    return !accepted && span.committed==span.freed_committed && whole.committed==whole.freed_committed ? 0:1;
}
