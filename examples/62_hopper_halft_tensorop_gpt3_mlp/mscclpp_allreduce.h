#include <cuda_fp16.h>

#include <cutlass/half.h>

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/nvls_device.hpp>
#include <mscclpp/semaphore.hpp>

constexpr size_t kMaxNumRanks = 8;

__device__ mscclpp::DeviceSyncer deviceSyncer;

// A semaphore for each rank except the local rank
__device__ mscclpp::SmDevice2DeviceSemaphoreDeviceHandle deviceSemaphores[kMaxNumRanks - 1];

// Barrier among all devices followed by a memory fence
// Should be called by all threads on all devices
// Assumes \p num_threads_per_block >= \p num_ranks
// Assumes \p num_ranks <= kMaxNumRanks
__device__ void memoryBarrier(
        int thread_id, int block_id, int num_threads_per_block, int num_blocks,
        int num_ranks) {
  // wait for every device
  if (block_id == 0) {
    // 1 less than the num_ranks because there is no semaphore for self
    if (thread_id < num_ranks - 1) {
      deviceSemaphores[thread_id].signal();
      deviceSemaphores[thread_id].wait();
    }
  }

  // wait for every thread in every block on this device
  deviceSyncer.sync(num_blocks);

  // memory fence across all threads on all devices
  __threadfence_system();
}

// -------------------------------------------
// Adapted from AllReduce6 in mscclpp repo
// Uses NVLS
// -------------------------------------------

// Assumes \p num_ranks <= kMaxNumRanks
__global__ void __launch_bounds__(1024, 1)
    mscclppAllReduceInplaceSum(
        cutlass::half_t* mc_ptr, size_t num_elements, 
        int my_rank, int num_ranks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads_per_block = blockDim.x;
  int num_blocks = gridDim.x;

  // start with a memory barrier to ensure all devices have written their values to their own memory
  memoryBarrier(tid, bid, num_threads_per_block, num_blocks, num_ranks);

  int rank_start = ((int64_t)num_elements * (int64_t)my_rank) / (int64_t)num_ranks;
  int rank_end = ((int64_t)num_elements * (int64_t)(my_rank + 1)) / (int64_t)num_ranks;

  constexpr int kVecSize = 8;
  int thread_offset = (bid * num_threads_per_block + tid) * kVecSize;
  int thread_step = (num_threads_per_block * num_blocks) * kVecSize; // number of threads * vector size

  for (int idx = rank_start + thread_offset; idx < rank_end; idx += thread_step) {
    uint4 val; // fits 8 cutlass::half_t elements; i.e., 4 half2 elements
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoad(val, (half2*)(mc_ptr + idx));
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (half2*)(mc_ptr + idx));
  }

  // end with a memory barrier to ensure all devices can now read their values from their own memory
  memoryBarrier(tid, bid, num_threads_per_block, num_blocks, num_ranks);
#endif
}
