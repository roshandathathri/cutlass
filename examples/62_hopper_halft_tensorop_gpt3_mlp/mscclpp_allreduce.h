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
__device__ void barrier(
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
}

// -------------------------------------------
// Adapted from AllReduce6 in mscclpp repo
// Uses NVLS
// -------------------------------------------

// Assumes \p num_ranks <= kMaxNumRanks
// Assumes \p kVecSize is 1, 2, 4, or 8 (default 8)
template <int kVecSize = 8>
__global__ void __launch_bounds__(1024, 1)
    mscclppNVLSAllReduceInplaceSum(
        cutlass::half_t* mc_ptr, size_t num_elements, 
        int my_rank, int num_ranks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads_per_block = blockDim.x;
  int num_blocks = gridDim.x;

  // start with a barrier to ensure all devices have written their values 
  // to their own memory (that is part of the multicast memory) 
  // before reading them in this kernel
  barrier(tid, bid, num_threads_per_block, num_blocks, num_ranks);

  // every device loads, reduces, and stores a partition of the multicast memory
  int rank_start = ((int64_t)num_elements * (int64_t)my_rank) / (int64_t)num_ranks;
  int rank_end = ((int64_t)num_elements * (int64_t)(my_rank + 1)) / (int64_t)num_ranks;

  int thread_offset = (bid * num_threads_per_block + tid) * kVecSize;
  int thread_step = (num_threads_per_block * num_blocks) * kVecSize; // number of threads * vector size

  for (int idx = rank_start + thread_offset; idx < rank_end; idx += thread_step) {
    if constexpr (kVecSize == 8) {
      uint4 val; // fits 8 cutlass::half_t elements; i.e., 4 half2 elements
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (half2*)(mc_ptr + idx));
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (half2*)(mc_ptr + idx));
    } else if constexpr (kVecSize == 4) {
      uint2 val; // fits 4 cutlass::half_t elements; i.e., 2 half2 elements
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (half2*)(mc_ptr + idx));
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (half2*)(mc_ptr + idx));
    } else if constexpr (kVecSize == 2) {
      uint1 val; // fits 2 cutlass::half_t elements; i.e., 1 half2 elements
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (half2*)(mc_ptr + idx));
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (half2*)(mc_ptr + idx));
    } else if constexpr (kVecSize == 1) {
      uint1 val; // fits 1 cutlass::half_t elements; i.e., 1 __half element
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (__half*)(mc_ptr + idx));
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (__half*)(mc_ptr + idx));
    }
  }

  // end with a barrier to ensure all devices can now read their values 
  // from their own memory (that is part of the multicast memory)
  // after writing them in this kernel
  barrier(tid, bid, num_threads_per_block, num_blocks, num_ranks);
#endif
}

// Assumes \p num_ranks <= kMaxNumRanks
__global__ void __launch_bounds__(1024, 1)
    mscclppNVLSAllReduceSum(
        cutlass::half_t* in_local_ptr, cutlass::half_t* out_mc_ptr, size_t num_elements, 
        int my_rank, int num_ranks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int num_threads_per_block = blockDim.x;
  int num_blocks = gridDim.x;

  // a barrier is not required as only local values are read in this kernel

  for (int step = 0; step < num_ranks; ++step) {
    // every device acts on a distinct partition in each step
    // this avoids two devices from trying to reduce 
    // the same element/address in the same step
    int64_t partition = (my_rank + step) % num_ranks;
    // every device reads all elements in its partition from its local memory 
    // and reduces it (partial value) to the multicast memory
    int part_start = ((int64_t)num_elements * partition) / (int64_t)num_ranks;
    int part_end = ((int64_t)num_elements * (partition + 1)) / (int64_t)num_ranks;

    constexpr int kVecSize = 8;
    int thread_offset = (bid * num_threads_per_block + tid) * kVecSize;
    int thread_step = (num_threads_per_block * num_blocks) * kVecSize; // number of threads * vector size

    for (int idx = part_start + thread_offset; idx < part_end; idx += thread_step) {
      uint4 val; // fits 8 cutlass::half_t elements; i.e., 4 half2 elements
      uint* uint_ptr = (uint*)(in_local_ptr + idx);
      val = {uint_ptr[0], uint_ptr[1], uint_ptr[2], uint_ptr[3]};
      mscclpp::DeviceMulticastPointerDeviceHandle::multimemStoreReduce(val, (half2*)(out_mc_ptr + idx));
    }
  }

  // end with a barrier to ensure all devices can now read their values 
  // from their own memory (that is part of the multicast memory)
  // after writing them in this kernel
  barrier(tid, bid, num_threads_per_block, num_blocks, num_ranks);
#endif
}
