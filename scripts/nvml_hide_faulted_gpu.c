/* LD_PRELOAD shim: hide a faulted trailing GPU from NVML.
 *
 * On the gpu-4090 box card3 (0000:CA:00.0, physical index 3 under
 * CUDA_DEVICE_ORDER=PCI_BUS_ID) intermittently falls off the bus, so
 * nvmlDeviceGetHandleByIndex(3) returns NVMLError_Unknown. Both NCCL
 * (misc/nvmlwrap.cc, via the system libnvidia-ml.so) and vLLM enumerate ALL
 * physical GPUs at init/import to build topology, so that one bad card aborts
 * every GRPO process regardless of CUDA_VISIBLE_DEVICES (NVML ignores it).
 *
 * Capping nvmlDeviceGetCount() makes every NVML consumer see only the leading
 * good cards (0..N-1), so the faulted trailing card is never probed. The hidden
 * card is never used for compute -- good cards are pinned via
 * CUDA_VISIBLE_DEVICES; it was only enumerated for topology.
 *
 * Build:  gcc -O2 -fPIC -shared -o nvml_hide_faulted_gpu.so \
 *             nvml_hide_faulted_gpu.c -ldl
 * Use:    LD_PRELOAD=/abs/path/nvml_hide_faulted_gpu.so <command>
 *         NVML_VISIBLE_COUNT overrides the cap (default 3).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>

typedef int nvmlReturn_t; /* NVML_SUCCESS == 0 */

static unsigned int visible_cap(void) {
    const char *e = getenv("NVML_VISIBLE_COUNT");
    return e ? (unsigned int)atoi(e) : 3u;
}

static void clamp(nvmlReturn_t r, unsigned int *count) {
    if (r == 0 && count) {
        unsigned int c = visible_cap();
        if (*count > c) *count = c;
    }
}

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
    static nvmlReturn_t (*real)(unsigned int *) = 0;
    if (!real) real = (nvmlReturn_t (*)(unsigned int *))dlsym(RTLD_NEXT, "nvmlDeviceGetCount_v2");
    nvmlReturn_t r = real(count);
    clamp(r, count);
    return r;
}

nvmlReturn_t nvmlDeviceGetCount(unsigned int *count) {
    static nvmlReturn_t (*real)(unsigned int *) = 0;
    if (!real) real = (nvmlReturn_t (*)(unsigned int *))dlsym(RTLD_NEXT, "nvmlDeviceGetCount");
    nvmlReturn_t r = real(count);
    clamp(r, count);
    return r;
}
