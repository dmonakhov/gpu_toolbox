/*
 * XID Fault Injection Tool
 *
 * Triggers various XID errors for testing GPU monitoring systems.
 * WARNING: This will cause GPU errors and may require GPU reset!
 *
 * Build:
 *   nvcc -o xid_inject xid_inject.cu
 *
 * Usage:
 *   ./xid_inject <xid_type>
 *
 * XID Types:
 *   31 - GPU memory page fault (invalid memory access)
 *   13 - Graphics exception (illegal instruction)
 *   43 - GPU timeout (infinite loop)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// XID 31: Access invalid GPU memory address
__global__ void trigger_xid31_page_fault() {
    // Dereference a NULL pointer - causes page fault
    int *invalid_ptr = (int*)0xDEADBEEF;
    *invalid_ptr = 42;
}

// XID 31 variant: Out of bounds access
__global__ void trigger_xid31_oob(int *data, size_t size) {
    // Access way beyond allocated memory
    size_t bad_idx = size + 1000000;
    data[bad_idx] = 123;
}

// XID 13: Illegal instruction (hard to trigger directly, this attempts it)
__global__ void trigger_xid13_illegal() {
    // Try to cause an illegal instruction by corrupting execution
    // This may or may not work depending on GPU architecture
    asm volatile("trap;");  // CUDA trap instruction
}

// XID 43: GPU timeout via infinite loop
__global__ void trigger_xid43_timeout() {
    // Infinite loop - will trigger GPU timeout
    // Note: Requires CUDA_LAUNCH_BLOCKING=1 or sync to trigger
    while(1) {
        // Prevent compiler optimization
        asm volatile("");
    }
}

void print_usage(const char *prog) {
    printf("XID Fault Injection Tool\n");
    printf("========================\n");
    printf("WARNING: This will cause GPU errors!\n\n");
    printf("Usage: %s <xid_type>\n\n", prog);
    printf("XID Types:\n");
    printf("  31    - GPU memory page fault (safest, recoverable)\n");
    printf("  31oob - Out of bounds memory access variant\n");
    printf("  13    - Graphics exception (trap instruction)\n");
    printf("  43    - GPU timeout (infinite loop - DANGEROUS)\n");
    printf("\nExample:\n");
    printf("  %s 31\n", prog);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Get GPU info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s (PCI %04x:%02x:%02x)\n",
           prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);

    const char *xid_type = argv[1];

    if (strcmp(xid_type, "31") == 0) {
        printf("Triggering XID 31 (page fault via NULL deref)...\n");
        printf("This should cause: NVRM: Xid: 31, GPU memory page fault\n\n");

        trigger_xid31_page_fault<<<1, 1>>>();
        cudaError_t err = cudaDeviceSynchronize();

        printf("CUDA error (expected): %s\n", cudaGetErrorString(err));
        printf("\nCheck dmesg for XID error:\n");
        printf("  sudo dmesg | tail -20 | grep -i xid\n");

    } else if (strcmp(xid_type, "31oob") == 0) {
        printf("Triggering XID 31 (page fault via out-of-bounds access)...\n");

        int *d_data;
        size_t size = 1024;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(int)));

        trigger_xid31_oob<<<1, 1>>>(d_data, size);
        cudaError_t err = cudaDeviceSynchronize();

        printf("CUDA error (expected): %s\n", cudaGetErrorString(err));
        cudaFree(d_data);

        printf("\nCheck dmesg for XID error:\n");
        printf("  sudo dmesg | tail -20 | grep -i xid\n");

    } else if (strcmp(xid_type, "13") == 0) {
        printf("Triggering XID 13 (graphics exception via trap)...\n");
        printf("This should cause: NVRM: Xid: 13, Graphics Exception\n\n");

        trigger_xid13_illegal<<<1, 1>>>();
        cudaError_t err = cudaDeviceSynchronize();

        printf("CUDA error (expected): %s\n", cudaGetErrorString(err));
        printf("\nCheck dmesg for XID error:\n");
        printf("  sudo dmesg | tail -20 | grep -i xid\n");

    } else if (strcmp(xid_type, "43") == 0) {
        printf("WARNING: This will cause GPU timeout and may hang!\n");
        printf("The GPU watchdog should kill it after ~2 seconds.\n");
        printf("Press Ctrl+C within 3 seconds to abort...\n");
        sleep(3);

        printf("\nTriggering XID 43 (timeout via infinite loop)...\n");
        printf("This should cause: NVRM: Xid: 43, GPU timeout\n\n");

        // Set a short timeout for testing
        // Note: actual timeout is controlled by driver
        trigger_xid43_timeout<<<1, 1>>>();
        cudaError_t err = cudaDeviceSynchronize();

        printf("CUDA error (expected): %s\n", cudaGetErrorString(err));
        printf("\nCheck dmesg for XID error:\n");
        printf("  sudo dmesg | tail -20 | grep -i xid\n");

    } else {
        fprintf(stderr, "Unknown XID type: %s\n\n", xid_type);
        print_usage(argv[0]);
        return 1;
    }

    // Try to reset the device
    printf("\nAttempting device reset...\n");
    cudaError_t reset_err = cudaDeviceReset();
    if (reset_err == cudaSuccess) {
        printf("Device reset successful.\n");
    } else {
        printf("Device reset failed: %s\n", cudaGetErrorString(reset_err));
        printf("You may need to run: sudo nvidia-smi --gpu-reset\n");
    }

    return 0;
}
