// Self-checking feature probe. See README.md for evidence scope and sources.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda/barrier>
#include <cuda/ptx>

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef PROBE_TARGET
#define PROBE_TARGET "unspecified"
#endif

constexpr int kTile = 256;
using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;

static std::string quote(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (c >= 32) out += static_cast<char>(c);
    else out += ' ';
  }
  return out + '"';
}

static void check(cudaError_t rc, const char* operation) {
  if (rc != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
}
#define CUDA_CHECK(expr) check((expr), #expr)

__global__ void thor_target_macros(int* values) {
#ifdef __CUDA_ARCH__
  values[0] = __CUDA_ARCH__;
#ifdef __CUDA_ARCH_SPECIFIC__
  values[1] = 1;
  values[2] = __CUDA_ARCH_SPECIFIC__;
#endif
#ifdef __CUDA_ARCH_FAMILY_SPECIFIC__
  values[3] = 1;
  values[4] = __CUDA_ARCH_FAMILY_SPECIFIC__;
#endif
#endif
}

__global__ void thor_simt_copy_reduce(const float* input, float* copied,
                                     float* sums, int rounds) {
  __shared__ float partial[kTile];
  const int lane = threadIdx.x;
  const int index = blockIdx.x * kTile + lane;
  float total = 0;
  for (int round = 0; round < rounds; ++round) {
    // Volatile preserves repeated global reads in this diagnostic workload.
    const float value = static_cast<const volatile float*>(input)[index]
                        + static_cast<float>(round & 1);
    copied[index] = value;
    partial[lane] = value;
    __syncthreads();
    for (int stride = kTile / 2; stride; stride /= 2) {
      if (lane < stride) partial[lane] += partial[lane + stride];
      __syncthreads();
    }
    if (lane == 0) total += partial[0];
    __syncthreads();
  }
  if (lane == 0) sums[blockIdx.x] = total;
}

__global__ void thor_tma_copy_reduce(const __grid_constant__ CUtensorMap tensor_map,
                                    float* copied, float* sums, int rounds) {
  __shared__ alignas(128) float tile[kTile];
  __shared__ float partial[kTile];
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ BlockBarrier ready;
  const int lane = threadIdx.x;
  const int index = blockIdx.x * kTile + lane;
  if (lane == 0) init(&ready, blockDim.x);
  __syncthreads();
  float total = 0;
  for (int round = 0; round < rounds; ++round) {
    BlockBarrier::arrival_token token;
    if (lane == 0) {
      const int32_t coords[1] = {static_cast<int32_t>(blockIdx.x * kTile)};
      cuda::ptx::cp_async_bulk_tensor(
          cuda::ptx::space_shared, cuda::ptx::space_global,
          tile, &tensor_map, coords, cuda::device::barrier_native_handle(ready));
      token = cuda::device::barrier_arrive_tx(ready, 1, sizeof(tile));
    } else {
      token = ready.arrive();
    }
    ready.wait(std::move(token));
    const float value = tile[lane] + static_cast<float>(round & 1);
    copied[index] = value;
    partial[lane] = value;
    __syncthreads();
    for (int stride = kTile / 2; stride; stride /= 2) {
      if (lane < stride) partial[lane] += partial[lane + stride];
      __syncthreads();
    }
    if (lane == 0) total += partial[0];
    // Complete generic shared-memory accesses before the next async overwrite.
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    __syncthreads();
  }
  if (lane == 0) sums[blockIdx.x] = total;
}

struct Attribute {
  const char* name;
  cudaDeviceAttr id;
};

static void print_attributes() {
  const Attribute attributes[] = {
      {"sm_count", cudaDevAttrMultiProcessorCount},
      {"l2_bytes", cudaDevAttrL2CacheSize},
      {"registers_per_sm", cudaDevAttrMaxRegistersPerMultiprocessor},
      {"registers_per_block", cudaDevAttrMaxRegistersPerBlock},
      {"threads_per_sm", cudaDevAttrMaxThreadsPerMultiProcessor},
      {"threads_per_block", cudaDevAttrMaxThreadsPerBlock},
      {"warp_size", cudaDevAttrWarpSize},
      {"blocks_per_sm", cudaDevAttrMaxBlocksPerMultiprocessor},
      {"shared_bytes_per_sm", cudaDevAttrMaxSharedMemoryPerMultiprocessor},
      {"shared_bytes_per_block_default", cudaDevAttrMaxSharedMemoryPerBlock},
      {"shared_bytes_per_block_optin", cudaDevAttrMaxSharedMemoryPerBlockOptin},
      {"reserved_shared_bytes_per_block", cudaDevAttrReservedSharedMemoryPerBlock},
      {"clock_rate_khz_device_attribute", cudaDevAttrClockRate},
      {"memory_clock_rate_khz_device_attribute", cudaDevAttrMemoryClockRate},
      {"memory_bus_width_bits", cudaDevAttrGlobalMemoryBusWidth},
      {"integrated", cudaDevAttrIntegrated},
      {"managed_memory", cudaDevAttrManagedMemory},
      {"concurrent_managed_access", cudaDevAttrConcurrentManagedAccess},
      {"cluster_launch", cudaDevAttrClusterLaunch},
  };
  bool first = true;
  std::cout << '{';
  for (const auto& a : attributes) {
    int value = 0;
    cudaError_t rc = cudaDeviceGetAttribute(&value, a.id, 0);
    if (!first) std::cout << ',';
    first = false;
    std::cout << quote(a.name) << ":{\"value\":";
    if (rc == cudaSuccess) std::cout << value;
    else std::cout << "null";
    std::cout << ",\"cuda_status\":" << static_cast<int>(rc) << '}';
  }
  std::cout << '}';
}

static void validate(const float* copied_device, const float* sums_device,
                     const std::vector<float>& input, int blocks, int rounds) {
  std::vector<float> copied(input.size()), sums(blocks);
  CUDA_CHECK(cudaMemcpy(copied.data(), copied_device, copied.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sums.data(), sums_device, sums.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < copied.size(); ++i) {
    const float expected = input[i] + static_cast<float>((rounds - 1) & 1);
    if (!std::isfinite(copied[i]) || copied[i] != expected)
      throw std::runtime_error("exact copy mismatch at index " + std::to_string(i));
  }
  for (int b = 0; b < blocks; ++b) {
    int expected = 0;
    for (int t = 0; t < kTile; ++t) expected += static_cast<int>(input[b * kTile + t]);
    expected = expected * rounds + kTile * (rounds / 2);
    if (!std::isfinite(sums[b]) || sums[b] != static_cast<float>(expected))
      throw std::runtime_error("exact reduction mismatch at block " + std::to_string(b));
  }
}

static int run(int argc, char** argv) {
  std::string mode = "check";
  int blocks = 4096, rounds = 32, repeats = 20;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) throw std::runtime_error("all arguments require a value");
    const std::string key = argv[i];
    if (key == "--mode") mode = argv[i + 1];
    else if (key == "--blocks") blocks = std::stoi(argv[i + 1]);
    else if (key == "--rounds") rounds = std::stoi(argv[i + 1]);
    else if (key == "--repeats") repeats = std::stoi(argv[i + 1]);
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (mode != "check" && mode != "timing" && mode != "profile")
    throw std::runtime_error("mode must be check, timing, or profile");
  if (blocks < 1 || blocks > 65536 || rounds < 1 || rounds > 256 ||
      repeats < 1 || repeats > 10000)
    throw std::runtime_error("blocks=1..65536, rounds=1..256, repeats=1..10000 required");
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  if (prop.major != 11 || prop.minor != 0)
    throw std::runtime_error("this validation campaign requires device CC 11.0");
  int runtime = 0, driver = 0;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  cudaFuncAttributes function_attributes[2]{};
  CUDA_CHECK(cudaFuncGetAttributes(&function_attributes[0], thor_simt_copy_reduce));
  CUDA_CHECK(cudaFuncGetAttributes(&function_attributes[1], thor_tma_copy_reduce));
  int* macros_device = nullptr;
  int macros[5] = {};
  CUDA_CHECK(cudaMalloc(&macros_device, sizeof(macros)));
  CUDA_CHECK(cudaMemset(macros_device, 0, sizeof(macros)));
  thor_target_macros<<<1, 1>>>(macros_device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(macros, macros_device, sizeof(macros), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(macros_device));
  if (macros[0] != 1100) throw std::runtime_error("unexpected device __CUDA_ARCH__");

  const size_t elements = static_cast<size_t>(blocks) * kTile;
  std::vector<float> input(elements);
  for (size_t i = 0; i < elements; ++i) input[i] = static_cast<int>(i % 17) - 8;
  float *input_device = nullptr, *copies[2]{}, *sums[2]{};
  CUDA_CHECK(cudaMalloc(&input_device, elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(input_device, input.data(), elements * sizeof(float),
                        cudaMemcpyHostToDevice));
  for (int kind = 0; kind < 2; ++kind) {
    CUDA_CHECK(cudaMalloc(&copies[kind], elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sums[kind], blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(copies[kind], 0xff, elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(sums[kind], 0xff, blocks * sizeof(float)));
  }
  CUtensorMap map{};
  const cuuint64_t dimensions[1] = {elements};
  const cuuint64_t unused_strides[1] = {elements * sizeof(float)};
  const cuuint32_t box[1] = {kTile}, element_strides[1] = {1};
  CUresult map_status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 1, input_device,
      dimensions, unused_strides, box, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (map_status != CUDA_SUCCESS)
    throw std::runtime_error("cuTensorMapEncodeTiled status " + std::to_string(map_status));
  auto launch = [&](int kind) {
    if (kind == 0)
      thor_simt_copy_reduce<<<blocks, kTile>>>(input_device, copies[0], sums[0], rounds);
    else
      thor_tma_copy_reduce<<<blocks, kTile>>>(map, copies[1], sums[1], rounds);
    CUDA_CHECK(cudaGetLastError());
  };

  // Profile mode captures precisely two named actions. It makes no event claim.
  if (mode == "profile") CUDA_CHECK(cudaProfilerStart());
  launch(0);
  launch(1);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (mode == "profile") CUDA_CHECK(cudaProfilerStop());
  validate(copies[0], sums[0], input, blocks, rounds);
  validate(copies[1], sums[1], input, blocks, rounds);

  std::vector<float> microseconds[2];
  if (mode == "timing") {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int kind = 0; kind < 2; ++kind) {
      for (int warmup = 0; warmup < 5; ++warmup) launch(kind);
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int trial = 0; trial < 5; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int repeat = 0; repeat < repeats; ++repeat) launch(kind);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        if (!std::isfinite(elapsed) || elapsed <= 0)
          throw std::runtime_error("invalid CUDA event elapsed time");
        microseconds[kind].push_back(elapsed * 1000.0f / repeats);
      }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    validate(copies[0], sums[0], input, blocks, rounds);
    validate(copies[1], sums[1], input, blocks, rounds);
  }
  for (int kind = 0; kind < 2; ++kind) {
    CUDA_CHECK(cudaFree(copies[kind]));
    CUDA_CHECK(cudaFree(sums[kind]));
  }
  CUDA_CHECK(cudaFree(input_device));

  std::cout << std::setprecision(9)
            << "{\"status\":\"passed\",\"mode\":" << quote(mode)
            << ",\"compiled_target\":" << quote(PROBE_TARGET)
            << ",\"device_name\":" << quote(prop.name)
            << ",\"compute_capability\":[" << prop.major << ',' << prop.minor << ']'
            << ",\"runtime_version\":" << runtime << ",\"driver_version\":" << driver
            << ",\"nvcc_version\":[" << __CUDACC_VER_MAJOR__ << ','
            << __CUDACC_VER_MINOR__ << ',' << __CUDACC_VER_BUILD__ << ']'
            << ",\"device_macros\":{\"cuda_arch\":" << macros[0]
            << ",\"architecture_specific_present\":" << (macros[1] ? "true" : "false")
            << ",\"architecture_specific_value\":" << macros[2]
            << ",\"family_specific_present\":" << (macros[3] ? "true" : "false")
            << ",\"family_specific_value\":" << macros[4] << '}'
            << ",\"device_attributes\":";
  print_attributes();
  std::cout << ",\"max_resident_warps_derived\":"
            << prop.maxThreadsPerMultiProcessor / prop.warpSize
            << ",\"clocks_and_power_changed\":false,\"power_watts\":null"
            << ",\"clock_note\":\"CUDA device attributes, not live clock telemetry\""
            << ",\"blocks\":" << blocks << ",\"threads\":" << kTile
            << ",\"rounds_per_kernel\":" << rounds << ",\"elements\":" << elements
            << ",\"validation\":\"finite and exact FP32 copy and block sums; no tolerance\""
            << ",\"kernels\":[";
  for (int kind = 0; kind < 2; ++kind) {
    if (kind) std::cout << ',';
    std::cout << "{\"name\":"
              << quote(kind == 0 ? "thor_simt_copy_reduce" : "thor_tma_copy_reduce")
              << ",\"checked_copy_elements\":" << elements
              << ",\"checked_block_sums\":" << blocks
              << ",\"function_attributes\":{\"registers_per_thread\":"
              << function_attributes[kind].numRegs
              << ",\"static_shared_bytes\":" << function_attributes[kind].sharedSizeBytes
              << ",\"local_bytes\":" << function_attributes[kind].localSizeBytes
              << ",\"max_dynamic_shared_bytes\":" << function_attributes[kind].maxDynamicSharedSizeBytes
              << ",\"max_threads_per_block\":" << function_attributes[kind].maxThreadsPerBlock
              << ",\"binary_version\":" << function_attributes[kind].binaryVersion
              << ",\"ptx_version\":" << function_attributes[kind].ptxVersion << '}'
              << ",\"event_mean_us_trials\":[";
    for (size_t i = 0; i < microseconds[kind].size(); ++i) {
      if (i) std::cout << ',';
      std::cout << microseconds[kind][i];
    }
    std::cout << "]}";
  }
  std::cout << "]}\n";
  return 0;
}

int main(int argc, char** argv) {
  try { return run(argc, argv); }
  catch (const std::exception& error) {
    std::cout << "{\"status\":\"error\",\"error\":" << quote(error.what()) << "}\n";
    return 1;
  }
}
