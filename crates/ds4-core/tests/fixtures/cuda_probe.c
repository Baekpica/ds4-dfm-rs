#include <stdlib.h>
#include <string.h>
int cuInit(unsigned flags) { return flags ? 1 : 0; }
int cuDeviceGet(int *device, int ordinal) {
    if (ordinal != 0) { return 1; }
    const char *visible = getenv("CUDA_VISIBLE_DEVICES");
    const char *order = getenv("CUDA_DEVICE_ORDER");
    if (visible && strncmp(visible, "GPU-111", 7) == 0) { *device = 1; return 0; }
    int index = visible ? atoi(visible) : 0;
    *device = order && strcmp(order, "PCI_BUS_ID") == 0 ? index : 1 - index;
    return 0;
}
int cuDeviceGetUuid_v2(unsigned char *uuid, int device) {
    memset(uuid, device ? 0x11 : 0, 16); return 0;
}
int cuDeviceGetAttribute(int *value, int attribute, int device) {
    (void)device;
    if (attribute != 18) { return 1; }
    *value = 0; return 0;
}
