/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example demonstrate a simple way to instantiate and run a TF32 GEMM using the new CUTLASS 3.0
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:

    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA) 
    which are more efficient than the Ampere tensor core instructions.

    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large 
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. Another advantage is that TMA can load in FP32 data and
    convert them implicitly to TF32.

    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).

    Examples:

      $ ./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm --m=2048 --n=2048 --k=2048
*/

#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>

#include <mscclpp/core.hpp>
#include <mpi.h>
#include <nccl.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include <mscclpp/nvls.hpp>
#include "mscclpp_allreduce.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementOutput           = cutlass::half_t;  // <- data type of elements in output matrix D
constexpr ncclDataType_t ElementNcclAllreduce 
                              = ncclHalf;         // <- data type for reducing elements in output matrix D

// Use wider type for reference kernel
using ElementReference        = float;
constexpr ncclDataType_t ElementNcclAllreduceReference 
                              = ncclFloat;

// The code section below describes matrix layout of input and output matrices. 
// All matrices are in ColumnMajor
using LayoutOutput          = cutlass::layout::ColumnMajor;

/////////////////////////////////////////////////////////////////////////////////////////////////

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  bool csv;

  int hidden_size;
  int batch_size;

  bool reference_check;
  bool use_nccl; // use NCCL instead of MSCCLPP
  bool inplace_mscclpp; // use MSCCLPP in-place allreduce
  int iterations;
  int mscclpp_block_size;
  int mscclpp_grid_size;

  int rank;
  int num_ranks;
  int num_devices_per_node;
  ncclComm_t nccl_comm;

  // MSCCLPP data structures
  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::shared_ptr<mscclpp::Communicator> communicator;
  std::vector<std::shared_ptr<mscclpp::Connection>> p2p_connections;
  std::vector<mscclpp::SmDevice2DeviceSemaphore> p2p_semaphores;
  std::shared_ptr<mscclpp::NvlsConnection> nvls_connection;

  Options(int argc, char const **args):
    help(false),
    csv(false),
    hidden_size(12288),
    batch_size(2048),
    reference_check(true),
    use_nccl(false),
    inplace_mscclpp(true),
    iterations(100),
    mscclpp_block_size(1024), 
    mscclpp_grid_size(8) { 

    // Initialize MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    // Set device
    setDevice();

    // Create NCCL communicator
    createNCCLComm();

    // Create MSCCL peer to peer connections
    createP2PConnections();

    // Parse command line to update hidden and batch size if needed
    parse(argc, args);

    // Now create NVLS Connection: requires output buffer size, which depends on hidden and batch sizes
    createNvlsConnection();
  }

  void setDevice() {
    // Get the number of GPUs per node
    cudaGetDeviceCount(&num_devices_per_node);

    // Prioritize locality: fill/use GPUs on the same node first
    CUDACHECK(cudaSetDevice(rank % num_devices_per_node));
  }

  // assumes one MPI process per GPU
  void createNCCLComm() {
    ncclUniqueId id;
    // generating NCCL unique ID at one process and broadcasting it to all
    if (rank == 0) {
      ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclCommInitRank(&nccl_comm, num_ranks, id, rank));
  }

  void createP2PConnections() {
    // Create a bootstrapped communicator
    bootstrap = std::shared_ptr<mscclpp::TcpBootstrap>(new mscclpp::TcpBootstrap(rank, num_ranks));
    bootstrap->initialize("127.0.0.1:50053");
    communicator = std::shared_ptr<mscclpp::Communicator>(new mscclpp::Communicator(bootstrap));

    // Create p2p connections
    mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
    mscclpp::EndpointConfig local_endpoint_config(transport);
    std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> future_connections;
    for (int remote_rank = 0; remote_rank < num_ranks; remote_rank++) {
      if (remote_rank != rank) {
        future_connections.emplace_back(communicator->connectOnSetup(remote_rank, 0, local_endpoint_config));
      }
    }

    // Setup the communicator with the p2p connections
    communicator->setup();

    // 1 less because there is no future_connections for the local rank
    for (int i = 0; i < num_ranks - 1; i++) {
      p2p_connections.emplace_back(future_connections[i].get());
      p2p_semaphores.emplace_back(mscclpp::SmDevice2DeviceSemaphore(*communicator, p2p_connections[i]));
    }

    // Setup the communicator again with the semaphores
    communicator->setup();

    std::vector<mscclpp::SmDevice2DeviceSemaphore::DeviceHandle> p2p_device_semaphores;
    // 1 less because there is no p2p_semaphores for the local rank
    for (int i = 0; i < num_ranks - 1; i++) {
      p2p_device_semaphores.emplace_back(p2p_semaphores[i].deviceHandle());
    }
    assert(num_ranks <= kMaxNumRanks);
    cudaMemcpyToSymbol(deviceSemaphores, p2p_device_semaphores.data(), 
                       sizeof(mscclpp::SmDevice2DeviceSemaphore::DeviceHandle) * (num_ranks - 1));
  }

  void createNvlsConnection() {
    // only supports 1 node currently
    assert(num_ranks == num_devices_per_node);
    size_t output_buffer_size = hidden_size * batch_size * sizeof(ElementOutput);

    if (rank == 0) {
      nvls_connection = std::make_shared<mscclpp::NvlsConnection>(output_buffer_size, num_devices_per_node);
         
      // output_buffer_size should be a multiple of the required minimum alignment
      // because the allocation is handled/managed by cutlass::DeviceAllocation
      assert(output_buffer_size % nvls_connection.getMultiCastMinGranularity() == 0);
      
      auto serialized = nvls_connection->serialize();
      for (int i = 1; i < num_ranks; i++) {
        MPICHECK(MPI_Send(serialized.data(), static_cast<int>(serialized.size()), MPI_BYTE, i, 0, MPI_COMM_WORLD));
      }
    } else {
      std::vector<char> serialized(output_buffer_size);
      MPICHECK(MPI_Recv(serialized.data(), static_cast<int>(serialized.size()), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      nvls_connection = std::make_shared<mscclpp::NvlsConnection>(serialized);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    nvls_connection->addDevice();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("csv")) {
      csv = true;
    }

    cmd.get_cmd_line_argument("hidden_size", hidden_size);
    cmd.get_cmd_line_argument("batch_size", batch_size);

    if (cmd.check_cmd_line_flag("do_not_verify")) {
      reference_check = false;
    }

    if (cmd.check_cmd_line_flag("use_nccl")) {
      use_nccl = true;
    } else if (cmd.check_cmd_line_flag("use_multicast_reduce")) {
      inplace_mscclpp = false;
    }

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("comm_block_size", mscclpp_block_size);
    cmd.get_cmd_line_argument("comm_grid_size", mscclpp_grid_size);
    assert(mscclpp_block_size >= num_ranks);
  }

  void print() {
    if (csv) {
      return;
    }

    if (rank != 0) {
      return;
    }

    std::cout << "Number of ranks: " << num_ranks << std::endl;
    std::cout << "Number of devices per node: " << num_devices_per_node << std::endl;

    printf("[%d,%d] HALFT tensor op AllReduce\n", \
      hidden_size, batch_size);

    fflush(stdout);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "61_ampere_halft_tensorop_gpt3_mlp example\n\n"
      << "  This example uses the CUTLASS Library to execute halft tensorop GPT3 MLP computation.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --hidden_size=<int>         Size of the hidden dimension (default 12288).\n\n"
      << "  --batch_size=<int>          Size of the batch dimension (default 2048).\n\n"
      << "  --csv                       If specified, prints in CSV format.\n\n"
      << "  --do_not_verify             If specified, skips verification of results using a reference implemenation.\n\n"
      << "  --use_nccl                  If specified, uses NCCL AllReduce instead of MSCCLPP implementation.\n\n"
      << "  --use_multicast_reduce      If specified, uses multicast reduce-based AllReduce with separate output memory.\n\n"
      << "  --comm_block_size           Number of threads per block to use in MSCCLPP AllReduce (default 1024).\n\n"
      << "  --comm_grid_size            Number of blocks per grid to use in MSCCLPP AllReduce (default 8).\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (default 100).\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./allreduce_bench --hidden_size=12288 --batch_size=1024 --csv\n\n";
    
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element>
struct ZeroFunc {
  struct Params {};
  Params params;
  CUTLASS_DEVICE
  ZeroFunc(Params const &params) : params(params) {}
  CUTLASS_DEVICE
  Element operator()() {
    return Element(0);
  }
};

template <typename WiderElement, typename Element>
__global__ void BlockCopyKernel(
  WiderElement *ptr_A,
  Element const *ptr_B,
  size_t capacity) {

  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (; idx < capacity; idx += gridDim.x * blockDim.x) {
    Element b = cutlass::ReferenceFactory<Element>::get(ptr_B, idx);
    cutlass::ReferenceFactory<WiderElement>::get(ptr_A, idx) = b;
  }
}

template <typename WiderElement, typename Element>
void BlockCopy(
  WiderElement *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  int grid_size = 0, 
  int block_size = 0,
  cudaStream_t stream = nullptr) {
  
  static_assert(sizeof(WiderElement) >= sizeof(Element), 
                  "WiderElement must be at least as wide as than Element");

  if (!grid_size || !block_size) {

    // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
    cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
      &grid_size,
      &block_size,
      reinterpret_cast<void const *>(BlockCopyKernel<WiderElement, Element>));

    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to query occupancy.");
    }

    // Limit block size. This has the effect of increasing the number of items processed by a
    // single thread and reduces the impact of initialization overhead.
    block_size = (block_size < 128 ? block_size : 128);
  }

  dim3 grid(grid_size, 1, 1);
  dim3 block(block_size, 1, 1);

  BlockCopyKernel<WiderElement, Element><<< grid, block, 0, stream >>>(
    ptr_A, 
    ptr_B, 
    capacity
  );
}

/// Adapted from cutlass/util/reference/device/tensor_compare.h
/// to support two types of elements
template <typename WiderElement, typename Element>
__global__ void BlockCompareRelativelyEqualKernel(
  int *equal, 
  WiderElement const *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  WiderElement epsilon,
  WiderElement nonzero_floor) {

  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  for (; idx < capacity; idx += gridDim.x * blockDim.x) {

    WiderElement a = cutlass::ReferenceFactory<WiderElement>::get(ptr_A, idx);
    WiderElement b = cutlass::ReferenceFactory<Element>::get(ptr_B, idx);

    if (!cutlass::relatively_equal(a, b, epsilon, nonzero_floor)) {
      *equal = 0;
      return;
    }
  }
}

/// Adapted from cutlass/util/reference/device/tensor_compare.h
/// to support two types of elements
template <typename WiderElement, typename Element>
bool BlockCompareRelativelyEqual(
  WiderElement const *ptr_A,
  Element const *ptr_B,
  size_t capacity,
  WiderElement epsilon,
  WiderElement nonzero_floor,
  int grid_size = 0, 
  int block_size = 0) {

  int equal_flag = 1;
  int *device_equal_flag = nullptr;

  if (cudaMalloc((void **)&device_equal_flag, sizeof(int)) != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device flag.");
  }

  if (cudaMemcpy(
    device_equal_flag, 
    &equal_flag, 
    sizeof(int), 
    cudaMemcpyHostToDevice) != cudaSuccess) {

    throw std::runtime_error("Failed to copy equality flag to device.");
  }

  if (!grid_size || !block_size) {

    // if grid_size or block_size are zero, query occupancy using the CUDA Occupancy API
    cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
      &grid_size,
      &block_size,
      reinterpret_cast<void const *>(BlockCompareRelativelyEqualKernel<WiderElement, Element>));

    if (result != cudaSuccess) {
      throw std::runtime_error("Failed to query occupancy.");
    }

    // Limit block size. This has the effect of increasing the number of items processed by a
    // single thread and reduces the impact of initialization overhead.
    block_size = (block_size < 128 ? block_size : 128);
  }

  dim3 grid(grid_size, 1, 1);
  dim3 block(block_size, 1, 1);

  BlockCompareRelativelyEqualKernel<WiderElement, Element><<< grid, block >>>(
    device_equal_flag, 
    ptr_A, 
    ptr_B, 
    capacity, 
    epsilon, 
    nonzero_floor
  );

  if (cudaMemcpy(
    &equal_flag, 
    device_equal_flag,
    sizeof(int), 
    cudaMemcpyDeviceToHost) != cudaSuccess) {
    
    cudaFree(device_equal_flag);

    throw std::runtime_error("Failed to copy equality flag from device.");
  }

  cudaFree(device_equal_flag);

  return equal_flag;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void InitializeMatrices(int hidden_size, int batch_size, 
  cutlass::DeviceAllocation<ElementOutput>* block_input,
  cutlass::DeviceAllocation<ElementOutput>* block_o,
  cutlass::DeviceAllocation<ElementReference>* block_ref_o, 
  ncclComm_t& nccl_comm) {

  block_input->reset(hidden_size * batch_size);
  block_o->reset(hidden_size * batch_size);
  block_ref_o->reset(hidden_size * batch_size);

  // Fill input matrices with uniform-distribution random data
  // For half_t tensors used in CUTLASS kernel, these ranges and precision for random values yield:
  // (1) 0.001f relatively equal results for hidden_size<=12k and batch_size<=2k
  // (2) 0.0001f relatively equal results for hidden_size<=1k and batch_size<=128
  cutlass::reference::device::BlockFillRandomUniform(
    block_input->get(), 
    block_input->size(), 
    2024, // seed
    ElementOutput{0.3}, 
    ElementOutput{-0.3}, 
    2);
  
  // Copy randomly generated values from input matrices to inputs for the reference kernel
  // The corresponding types of the tensors in the two kernels might be different
  assert(block_input->size() == block_o->size());
  BlockCopy(block_o->get(), block_input->get(), block_input->size());
  assert(block_input->size() == block_ref_o->size());
  BlockCopy(block_ref_o->get(), block_input->get(), block_input->size());
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int run(Options &options) {
  int rank = options.rank;
  int num_ranks = options.num_ranks;
  ncclComm_t& nccl_comm = options.nccl_comm;
  int hidden_size = options.hidden_size;
  int batch_size = options.batch_size;

  options.print();

  // Create stream on this process/GPU
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  //
  // Data members
  //
  // Create matrices on this process/GPU
  cutlass::DeviceAllocation<ElementOutput> block_input;
  cutlass::DeviceAllocation<ElementOutput> block_o;
  // Create matrices on this process/GPU for the reference kernel
  cutlass::DeviceAllocation<ElementReference> block_ref_o;

  // Initialize matrices on this process/GPU
  InitializeMatrices(hidden_size, batch_size, &block_input, &block_o, &block_ref_o, nccl_comm);

  // Create the multicast pointer for the output tensor
  std::shared_ptr<char> block_o_mc = options.nvls_connection->bindAllocatedCuda(block_o.getHandle(), block_o.bytes_allocated());
  ElementOutput* block_o_mc_ptr = (ElementOutput*)block_o_mc.get();

  // Result structure
  Result result;

  constexpr size_t kNumEventsPerIteration = 2;

  //
  // Construct events
  //
  std::vector<cudaEvent_t> events;
  events.resize(kNumEventsPerIteration * options.iterations);
  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  // TODO(roshan) skip reference check if reference_check = false
  // Launch all-reduce on this process/GPU
  NCCLCHECK(ncclAllReduce((const void*)block_ref_o.get(), (void*)block_ref_o.get(), 
        block_ref_o.size(), ElementNcclAllreduceReference, ncclSum, nccl_comm, 
        0)); // use the default stream

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // synchronize all devices before beginning profiling
  MPI_Barrier(MPI_COMM_WORLD);

  //
  // Run profiling loop
  //
  for (int iter = 0; iter < options.iterations; ++iter) {
    if (options.use_nccl || options.inplace_mscclpp) {
      // reset the input values at the beginning of each iteration
      BlockCopy(block_o.get(), block_input.get(), block_input.size(), 0, 0, stream);
    }

    // Record an event at the start of the all-reduce
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 0], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch all-reduce on each device on this process/GPU
    if (options.use_nccl) {
      NCCLCHECK(ncclAllReduce((const void*)block_o.get(), (void*)block_o.get(), 
            block_o.size(), ElementNcclAllreduce, ncclSum,
            nccl_comm, stream));
    } else if (options.inplace_mscclpp) {
      int num_threads_per_block = options.mscclpp_block_size;
      int num_blocks = options.mscclpp_grid_size;
      mscclppAllReduceInplaceSum<<<num_blocks, num_threads_per_block, 0, stream>>>(block_o_mc_ptr, block_o.size(),
                                                                                   rank, num_ranks);
    } else {
      int num_threads_per_block = options.mscclpp_block_size;
      int num_blocks = options.mscclpp_grid_size;
      assert(block_input.size() == block_o.size());
      mscclppAllReduceSum<<<num_blocks, num_threads_per_block, 0, stream>>>(block_input.get(), block_o_mc_ptr, block_o.size(),
                                                                            rank, num_ranks);
    }    

    // Record an event when the all-reduce is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 1], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }
  //
  // Stop profiling loop
  //

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[options.iterations * kNumEventsPerIteration - 1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed time
  float comm_time_ms{0.0f};
  float time_ms;
  for (int iter = 0; iter < options.iterations; ++iter) {
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 0], 
      events[iter * kNumEventsPerIteration + 1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    comm_time_ms += time_ms;
  }
  comm_time_ms /= static_cast<float>(options.iterations);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  // Then check if output from all-reduce kernel and reference kernel are equal or not
  assert(block_o.size() == block_ref_o.size());
  // This is the maximum relative error we expect from the relative equality check
  // A higher value can lead to errors due to quantized/lower-precision values 
  // that are used in the all-reduce kernel
  ElementReference epsilon(0.001f);
  ElementReference nonzero_floor(std::numeric_limits<ElementReference>::min());
  bool passed = BlockCompareRelativelyEqual(
    block_ref_o.get(), block_o.get(), block_o.size(), epsilon, nonzero_floor);

  MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

  if (passed) {
    int64_t data_size_bytes = int64_t{options.hidden_size} * options.batch_size * 2;
    float comm_time_us = comm_time_ms * 1000;
    float bandwidth_gbps = (data_size_bytes * 1000000.0f) / comm_time_us / 1024 / 1024 / 1024;

    if (options.csv) {
      if (rank == 0) {
        std::cout << "Hidden size,Batch size,Data size (bytes),#Ranks,#DevicesPerNode,Comm. Time (us), Bandwidth (GiB/s)" << std::endl;
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << options.hidden_size << ",";
        os << options.batch_size << ",";
        os << data_size_bytes << ",";
        os << num_ranks << ",";
        os << options.num_devices_per_node << ",";
        os << comm_time_us << ",";
        os << bandwidth_gbps << "\n";
        std::cout << os.str();
      }
    } else {
      if (rank == 0) {
        std::cout << "Hidden size\t| Batch size\t| Data size\t| Rank\t| Comm. Time (us)\t| Bandwidth (GiB/s)" << std::endl;
      }
    
      MPI_Barrier(MPI_COMM_WORLD);
    
      std::ostringstream os;
      os << std::fixed << std::setprecision(3);
      os << options.hidden_size << "\t\t| ";
      os << options.batch_size << "\t\t| ";
      os << data_size_bytes << "\t| ";
      os << rank << "\t| ";
      os << comm_time_us << "\t\t| ";
      os << bandwidth_gbps << "\n";
      std::cout << os.str();

      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (!options.csv && rank == 0) {
    std::cout << (passed ? "Passed" : "Failed") << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return (passed ? 0  : -1);
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

int main(int argc, const char **argv) {
  
  // Must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major < 9) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture or "
      << "later (compute capability 90 or greater).\n";
    return 0;
  }

  //
  // Parse options
  //

  Options options(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return run(options);
#else
  return 0;
#endif
}
