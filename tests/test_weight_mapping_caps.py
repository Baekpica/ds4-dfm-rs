#!/usr/bin/env python3
"""CPU controls for actual CUDA capability selection; no simulated GPU qualification."""
from pathlib import Path
import subprocess,tempfile
repo=Path(__file__).resolve().parents[1];s=(repo/'ds4_cuda.cu').read_text()
def function(name):
 a=s.index('static ',s.index(name)-30);b=s.index('{',a);depth=1;e=b+1
 while depth:
  depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[a:e]
code='''
#include <cstdio>
const int cudaSuccess=0,cudaHostRegisterMapped=2,cudaHostRegisterReadOnly=8;
const int cudaDevAttrHostRegisterReadOnlySupported=113,cudaDevAttrPageableMemoryAccess=88,cudaDevAttrPageableMemoryAccessUsesHostPageTables=100;
static int ro=0,pageable=0,tables=0,query_error=0;
int cudaGetDevice(int *d){*d=0;return 0;}
int cudaDeviceGetAttribute(int *v,int attr,int d){(void)d;if(query_error)return 1;*v=attr==113?ro:attr==88?pageable:tables;return 0;}
'''+function('cuda_model_host_register_flags')+'\n'+function('cuda_model_coherent_host_access')+'''
int main(){
 bool ok=cuda_model_host_register_flags()==2;ro=1;ok=ok && cuda_model_host_register_flags()==10;
 ok=ok && !cuda_model_coherent_host_access(0);pageable=1;ok=ok && !cuda_model_coherent_host_access(0);
 tables=1;ok=ok && cuda_model_coherent_host_access(0);pageable=0;ok=ok && !cuda_model_coherent_host_access(0);
 pageable=1;query_error=1;ok=ok && !cuda_model_coherent_host_access(0) && cuda_model_host_register_flags()==2;
 printf("{\\\"capability_controls_passed\\\":%s,\\\"scope\\\":\\\"production-functions-with-CPU-device-attribute-controls\\\",\\\"spark_pointer_path_qualified\\\":false}\\n",ok?"true":"false");return ok?0:1;
}
'''
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'test.cc').write_text(code);subprocess.run(['c++','-O2',str(p/'test.cc'),'-o',str(p/'test')],check=True)
 raise SystemExit(subprocess.run([str(p/'test')]).returncode)
