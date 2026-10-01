// Bounded actual-checkpoint mapping probe; never copies a model payload.
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
__global__ void mapped_read(unsigned char *out,const unsigned char *in) {
    if(threadIdx.x<64)out[threadIdx.x]=in[threadIdx.x];
}
int main(int argc,char **argv) {
    if(argc<3 || argc>4)return 2;
    int dev=0;cudaGetDevice(&dev);cudaDeviceProp p;cudaGetDeviceProperties(&p,dev);
    int ro=0,ats=0,reg=0;cudaDeviceGetAttribute(&ro,cudaDevAttrHostRegisterReadOnlySupported,dev);
    cudaDeviceGetAttribute(&ats,cudaDevAttrPageableMemoryAccessUsesHostPageTables,dev);
    cudaDeviceGetAttribute(&reg,cudaDevAttrHostRegisterSupported,dev);
    const bool writable=argc==4 && !strcmp(argv[3],"writable");
    const bool automatic=argc==4 && !strcmp(argv[3],"auto");
    const size_t length=1u<<20;const uint64_t tensor_offset=strtoull(argv[2],nullptr,10);const off_t off=tensor_offset&~4095ULL;
    const size_t delta=tensor_offset-off;
    int fd=open(argv[1],O_RDONLY);if(fd<0)return 2;
    unsigned char *map=(unsigned char *)mmap(nullptr,length,PROT_READ | (writable ? PROT_WRITE : 0),MAP_PRIVATE,fd,off);
    if(map==MAP_FAILED)return 2;
    unsigned flags=cudaHostRegisterMapped | (automatic && ro ? cudaHostRegisterReadOnly : 0u);
#ifdef DS4_MAPPING_USE_READONLY
    flags|=cudaHostRegisterReadOnly;
#endif
    cudaError_t e=cudaHostRegister(map,length,flags);const char *error=cudaGetErrorString(e);
    unsigned char *gpu=nullptr,*output=nullptr;unsigned char got[64]={0};bool equal=false;
    if(e==cudaSuccess && cudaHostGetDevicePointer((void **)&gpu,map,0)==cudaSuccess && cudaMalloc(&output,64)==cudaSuccess) {
        mapped_read<<<1,64>>>(output,gpu+delta);cudaDeviceSynchronize();
        if(cudaMemcpy(got,output,64,cudaMemcpyDeviceToHost)==cudaSuccess)equal=memcmp(got,map+delta,64)==0;
        cudaFree(output);cudaHostUnregister(map);
    }
    printf("{\"gpu\":\"%s\",\"scope\":\"one-MiB-real-Q6-embedding-file-map\",\"map_alignment\":%llu,\"file_offset\":%llu,\"device_read_file_offset\":%llu,\"mapped_bytes\":%zu,\"flags\":%u,\"readonly_supported\":%d,\"host_page_tables\":%d,\"host_register_supported\":%d,\"cuda_error\":\"%s\",\"exact_device_read\":%s,\"canonical_device_copy_bytes\":0}\n",p.name,(unsigned long long)((uintptr_t)map%4096),(unsigned long long)off,(unsigned long long)tensor_offset,length,flags,ro,ats,reg,error,equal?"true":"false");
    munmap(map,length);close(fd);return equal?0:1;
}
