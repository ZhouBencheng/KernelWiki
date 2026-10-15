// Compile-only target gate. Never launch this as a feature correctness test.
extern "C" __global__ void thor_tcgen05_fence_compile_only() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
