/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <iostream>
#include <algorithm>
#include <vector>
#include <iomanip>

#include <mpi.h>
#include <nccl.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/relatively_equal.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_relu.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

#ifndef BATCH_SIZE
#define HIDDEN_SIZE 12288
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// This section selects tuned hyperparameters specific to the batch size
// that is chosen at compile-time 
// (hidden size is assumed to be 12288)
// (only selected batch sizes are supported: unsupported ones default to a batch size of 1)

#if BATCH_SIZE == 2048

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 3;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 1792

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<160, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<80, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 1536

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 3;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 1280

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<160, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<80, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 3;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 1024

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 3;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 512

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 3;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 256

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 128, 32>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 3;
// Split K dimension into 1 partitions
constexpr int SplitKSlices1 = 1;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 128

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<256, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<256, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<64, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<64, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 1;

#elif BATCH_SIZE == 64

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#elif BATCH_SIZE == 32

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#elif BATCH_SIZE == 16

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#elif BATCH_SIZE == 8

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#elif BATCH_SIZE == 4

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#elif BATCH_SIZE == 2

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#else 

#define BATCH_SIZE 1

// This code section describes the tile size <M, N, K> a thread block will compute
using ShapeMMAThreadBlock1 =
    cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeMMAThreadBlock2 =
    cutlass::gemm::GemmShape<64, 64, 64>;
// This code section describes tile size <M, N, K> a warp will compute
using ShapeMMAWarp1 = cutlass::gemm::GemmShape<32, 32, 64>;
using ShapeMMAWarp2 = cutlass::gemm::GemmShape<32, 32, 64>;
// This code section describes the size <M, N, K> of MMA op
using ShapeMMAOp1 = cutlass::gemm::GemmShape<16, 8, 16>;
using ShapeMMAOp2 = cutlass::gemm::GemmShape<16, 8, 16>;
// Number of pipelines you want to use
constexpr int NumStages1 = 4;
constexpr int NumStages2 = 4;
// Split K dimension into 2 partitions
constexpr int SplitKSlices1 = 2;
constexpr int SplitKSlices2 = 2;

#endif


///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;               // <- data type of accumulator
using ElementComputeEpilogue = cutlass::half_t; // <- data type of epilogue operations
using ElementInputA1 = cutlass::half_t;          // <- data type of elements in input matrix A
using ElementInputB1 = cutlass::half_t;          // <- data type of elements in input matrix B
using ElementInputA2 = cutlass::half_t;         // <- data type of elements in input matrix A2
using ElementIntermediateB2 = cutlass::half_t;  // <- data type of elements in intermediate matrix B2
using ElementOutput = cutlass::half_t;          // <- data type of elements in output matrix D
constexpr ncclDataType_t ElementNcclAllreduce 
                              = ncclHalf;    // <- data type for reducing elements in output matrix D

// Use wider type for reference kernel
using ElementReference = float;
constexpr ncclDataType_t ElementNcclAllreduceReference 
                              = ncclFloat;

// The code section below describes matrix layout of input and output matrices. 
#if BATCH_SIZE >= 8
// All matrices are in RowMajor
using LayoutInputA1 = cutlass::layout::RowMajor;
using LayoutInputB1 = cutlass::layout::RowMajor;
using LayoutInputA2 = cutlass::layout::RowMajor;
using LayoutIntermediateB2 = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;
#else
// Only A matrices are in RowMajor; the other matrices are in ColumnMajor
// (this is to avoid cutlass error: Error Misaligned Operand at: 876)
using LayoutInputA1 = cutlass::layout::RowMajor;
using LayoutInputB1 = cutlass::layout::ColumnMajor;
using LayoutInputA2 = cutlass::layout::RowMajor;
using LayoutIntermediateB2 = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;
#endif

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

  using Gemm1 = cutlass::gemm::device::Gemm<ElementInputA1,
                                          LayoutInputA1,
                                          ElementInputB1,
                                          LayoutInputB1,
                                          ElementIntermediateB2,
                                          LayoutIntermediateB2,
                                          ElementAccumulator,
                                          MMAOp,
                                          SmArch,
                                          ShapeMMAThreadBlock1,
                                          ShapeMMAWarp1,
                                          ShapeMMAOp1,
                                          EpilogueOp,
                                          SwizzleThreadBlock,
                                          NumStages1, 
                                          8,
                                          8,
                                          true>;

using Gemm2 = cutlass::gemm::device::Gemm<ElementInputA2,
                                         LayoutInputA2,
                                         ElementIntermediateB2,
                                         LayoutIntermediateB2,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock2,
                                         ShapeMMAWarp2,
                                         ShapeMMAOp2,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages2,
                                         8,
                                         8,
                                         true>;

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
  int iterations;

  int rank;
  int num_ranks;
  int num_devices_per_node;
  ncclComm_t nccl_comm;

  cutlass::gemm::GemmCoord problem_size1;
  cutlass::gemm::GemmCoord problem_size2;

  Options():
    help(false),
    csv(false),
    hidden_size(HIDDEN_SIZE),
    batch_size(BATCH_SIZE),
    reference_check(true),
    iterations(20) { 

    // Initialize MPI
    MPICHECK(MPI_Init(NULL, NULL));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &num_ranks));

    // Set device
    setDevice();

    // Create NCCL communicator
    createNCCLComm();

    // Now initialize problem sizes
    init_problem_sizes();
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

    if (cmd.check_cmd_line_flag("do_not_verify")) {
      reference_check = false;
    }

    cmd.get_cmd_line_argument("iterations", iterations);
  }

  void init_problem_sizes() {
    problem_size1.m() = 4 * hidden_size / num_ranks;
    problem_size1.n() = batch_size;
    problem_size1.k() = hidden_size;

    problem_size2.m() = hidden_size;
    problem_size2.n() = batch_size;
    problem_size2.k() = 4 * hidden_size / num_ranks;
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

    printf("[%d,%d] x [%d,%d] HALFT tensor op Matrix Multiply\n", \
      problem_size1.m(), problem_size1.k(), 
      problem_size1.k(), problem_size1.n());

    printf("[%d,%d] HALFT tensor op ReLU\n", \
      problem_size1.m(),  problem_size1.n());

    printf("[%d,%d] x [%d,%d] HALFT tensor op Matrix Multiply\n", \
      problem_size2.m(), problem_size2.k(), 
      problem_size2.k(), problem_size2.n());

    printf("[%d,%d] HALFT tensor op AllReduce\n", \
      problem_size2.m(),  problem_size2.n());
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "61_ampere_halft_tensorop_gpt3_mlp example\n\n"
      << "  This example uses the CUTLASS Library to execute halft tensorop GPT3 MLP computation.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --csv                       If specified, prints in CSV format.\n\n"
      << "  --do_not_verify             If specified, skips verification of results using a reference implemenation.\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/61_ampere_halft_tensorop_gpt3/61_ampere_halft_tensorop_2mlp_gpt3_mlp_h<hidden_size>_b<batch_size>\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas1 = problem_size1.product();
    int64_t fmas2 = problem_size2.product();

    // Two flops per multiply-add
    return 2.0 * double(fmas1 + fmas2) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
                                         
template<class T>
void memset_random(T* f, int numVals, T* values, size_t nelems)
{
  assert(f != nullptr);
  for (size_t i = 0; i < nelems; i++) {
    f[i] =  values[rand()%numVals];
  }
}
                                         
template<class Tdst, class Tsrc>
void copy(Tdst* dst, Tsrc* src, size_t nelems)
{
  assert(src != nullptr);
  assert(dst != nullptr);
  for (size_t i = 0; i < nelems; i++) {
    dst[i] =  src[i];
  }
}
                                         
template<class Tdst, class Tsrc>
bool relatively_equal_array(Tdst* dst, Tsrc* src, size_t nelems, bool print_error)
{
  assert(src != nullptr);
  assert(dst != nullptr);

  using WidestType = float;
  // This is the maximum relative error we expect from the relative equality check
  // A higher value can lead to errors due to quantized/lower-precision values 
  // that are used in the CUTLASS kernel
  WidestType epsilon(0.001f);
  WidestType nonzero_floor(std::numeric_limits<WidestType>::min());

  size_t nerrors = 0;
  for (size_t i = 0; i < nelems; i++) {
    WidestType d = dst[i];
    WidestType s = src[i];
    if (cutlass::relatively_equal(d, s, epsilon, nonzero_floor) == false) {
      if (print_error && nerrors < 10) {
        printf("i=%zu, d=%f, s=%f\n", i, d, s);
      }
      nerrors++;
    }
  }

  if (print_error && nerrors > 0) {
    float perc_errors = 
      (static_cast<float>(nerrors) * 100.0f) / static_cast<float>(nelems);
    printf("nerrors=%zu, nelems=%zu, perc_errors=%.2f\n", 
            nerrors, nelems, perc_errors);
  }
  return nerrors == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void initializeMatrices(cutlass::gemm::GemmCoord& problem_size1, 
  cutlass::gemm::GemmCoord& problem_size2, 
  cutlass::HostTensor<ElementInputA1, LayoutInputA1>* tensor_a, 
  cutlass::HostTensor<ElementReference, LayoutInputA1>* tensor_ref_a, 
  cutlass::HostTensor<ElementInputB1, LayoutInputB1>* tensor_b, 
  cutlass::HostTensor<ElementReference, LayoutInputB1>* tensor_ref_b, 
  cutlass::HostTensor<ElementInputA2, LayoutInputA2>* tensor_a2, 
  cutlass::HostTensor<ElementReference, LayoutInputA2>* tensor_ref_a2, 
  cutlass::HostTensor<ElementIntermediateB2, LayoutIntermediateB2>* tensor_b2, 
  cutlass::HostTensor<ElementReference, LayoutIntermediateB2>* tensor_ref_b2, 
  cutlass::HostTensor<ElementOutput, LayoutOutput>* tensor_o,
  cutlass::HostTensor<ElementReference, LayoutOutput>* tensor_ref_o) {

  // Initialize tensors using CUTLASS helper functions
  tensor_a->resize(
    problem_size1.mk());  // <- Create matrix A for first GEMM with dimensions M x K
                          // CUTLASS kernel
  tensor_ref_a->resize(
    problem_size1.mk());  // <- Create matrix A for first GEMM with dimensions M x K
                          // reference kernel
  tensor_b->resize(
      problem_size1.kn());  // <- Create matrix B for first GEMM with dimensions K x N
                          // CUTLASS kernel
  tensor_ref_b->resize(
      problem_size1.kn());  // <- Create matrix B for first GEMM with dimensions K x N
                          // reference kernel
  tensor_a2->resize(
      problem_size2.mk());  // <- Create matrix A for second GEMM with dimensions M x K
                          // CUTLASS kernel
  tensor_ref_a2->resize(
      problem_size2.mk());  // <- Create matrix A for second GEMM with dimensions M x K
                          // reference kernel
  tensor_b2->resize(
      problem_size2.kn());  // <- Create matrix B for second GEMM with dimensions K x N
                          // CUTLASS kernel
  assert(problem_size1.mn() == problem_size2.kn());
  tensor_ref_b2->resize(problem_size2.kn());
                          // reference kernel
  tensor_o->resize(
      problem_size2.mn());  // <- Create matrix O for second GEMM with dimensions M x N used to store output
                          // CUTLASS kernel
  tensor_ref_o->resize(
      problem_size2.mn());  // <- Create matrix O for second GEMM with dimensions M x N used to store output
                          // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions;
  // For half_t tensors used in CUTLASS kernel, these ranges and precision for random values yield:
  // (1) 0.001f relatively equal results for hidden_size<=12k and batch_size<=2k
  // (2) 0.0001f relatively equal results for hidden_size<=1k and batch_size<=128
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a->host_view(),
      1,
      ElementInputA1(0.3),
      ElementInputA1(-0.3),
      2);  // <- Fill matrix A on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b->host_view(),
      1,
      ElementInputB1(0.3),
      ElementInputB1(-0.3),
      2);  // <- Fill matrix B on host with uniform-distribution random data
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a2->host_view(),
      1,
      ElementInputA2(0.3),
      ElementInputA2(-0.3),
      2);  // <- Fill matrix A2 on host with uniform-distribution random data

  // Copy input tensors from that of CUTLASS kernel to that of reference kernel
  // The corresponding types of the tensors in the two kernels might be different
  copy(tensor_ref_a->host_data(), tensor_a->host_data(), tensor_a->size());
  copy(tensor_ref_b->host_data(), tensor_b->host_data(), tensor_b->size());
  copy(tensor_ref_a2->host_data(), tensor_a2->host_data(), tensor_a2->size());
          
  cutlass::reference::host::TensorFill(
      tensor_b2->host_view());  // <- fill matrix B2 on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_b2->host_view());  // <- fill matrix B2 on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_o->host_view());  // <- fill matrix O on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_o->host_view());  // <- fill matrix O for reference on host with zeros

  // Copy data from host to GPU
  tensor_a->sync_device();
  tensor_ref_a->sync_device();
  tensor_b->sync_device();
  tensor_ref_b->sync_device();
  tensor_a2->sync_device();
  tensor_ref_a2->sync_device();
  tensor_b2->sync_device();
  tensor_ref_b2->sync_device();
  tensor_o->sync_device();
  tensor_ref_o->sync_device();
}

void constructOps(cutlass::gemm::GemmCoord& problem_size1, 
  cutlass::gemm::GemmCoord& problem_size2, 
  cutlass::HostTensor<ElementInputA1, LayoutInputA1>& tensor_a, 
  cutlass::HostTensor<ElementInputB1, LayoutInputB1>& tensor_b, 
  cutlass::HostTensor<ElementInputA2, LayoutInputA2>& tensor_a2, 
  cutlass::HostTensor<ElementIntermediateB2, LayoutIntermediateB2>& tensor_b2, 
  cutlass::HostTensor<ElementOutput, LayoutOutput>& tensor_o, 
  Gemm1* gemm_op1,
  Gemm2* gemm_op2, 
  cutlass::device_memory::allocation<uint8_t>* workspace1, 
  cutlass::device_memory::allocation<uint8_t>* workspace2) {

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm1::Arguments arguments1{problem_size1,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_b2.device_ref(),  // does NOT matter because beta = 0
                                     tensor_b2.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     SplitKSlices1};        // <- k-dimension split factor
  typename Gemm2::Arguments arguments2{problem_size2,  // <- problem size of matrix multiplication
                                     tensor_a2.device_ref(),  // <- reference to matrix A on device
                                     tensor_b2.device_ref(),  // <- reference to matrix B on device
                                     tensor_o.device_ref(),  // does NOT matter because beta = 0
                                     tensor_o.device_ref(),  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     SplitKSlices2};        // <- k-dimension split factor

  cutlass::Status status;

  // Check the problem size is supported or not 
  status = gemm_op1->can_implement(arguments1);
  CUTLASS_CHECK(status);
  status = gemm_op2->can_implement(arguments2);
  CUTLASS_CHECK(status);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size1 = Gemm1::get_workspace_size(arguments1);
  size_t workspace_size2 = Gemm2::get_workspace_size(arguments2);

  // Allocate workspace memory
  workspace1->reset(workspace_size1);
  workspace2->reset(workspace_size2);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op1->initialize(arguments1, workspace1->get());
  CUTLASS_CHECK(status);
  status = gemm_op2->initialize(arguments2, workspace2->get());
  CUTLASS_CHECK(status);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int run(Options &options) {
  int rank = options.rank;
  int num_ranks = options.num_ranks;
  ncclComm_t& nccl_comm = options.nccl_comm;

  options.print();

  // Create stream on this process/GPU
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size1 = options.problem_size1;
  cutlass::gemm::GemmCoord problem_size2 = options.problem_size2;

  // Create matrices on this process/GPU
  cutlass::HostTensor<ElementInputA1, LayoutInputA1> tensor_a;
  cutlass::HostTensor<ElementReference, LayoutInputA1> tensor_ref_a;
  cutlass::HostTensor<ElementInputB1, LayoutInputB1> tensor_b;
  cutlass::HostTensor<ElementReference, LayoutInputB1> tensor_ref_b;
  cutlass::HostTensor<ElementInputA2, LayoutInputA2> tensor_a2;
  cutlass::HostTensor<ElementReference, LayoutInputA2> tensor_ref_a2;
  cutlass::HostTensor<ElementIntermediateB2, LayoutIntermediateB2> tensor_b2;
  cutlass::HostTensor<ElementReference, LayoutIntermediateB2> tensor_ref_b2;
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_o;
  cutlass::HostTensor<ElementReference, LayoutOutput> tensor_ref_o;

  // Initialize matrices on this process/GPU
  initializeMatrices(problem_size1, problem_size2, 
    &tensor_a, &tensor_ref_a, 
    &tensor_b, &tensor_ref_b, 
    &tensor_a2, &tensor_ref_a2, 
    &tensor_b2, &tensor_ref_b2, 
    &tensor_o, &tensor_ref_o);

  // Create the GEMM ops
  Gemm1 gemm_op1;
  Gemm2 gemm_op2;
  // Create the workspaces for the GEMM ops
  cutlass::device_memory::allocation<uint8_t> workspace1;
  cutlass::device_memory::allocation<uint8_t> workspace2;

  // Construct the GEMM op on this process/GPU
  constructOps(problem_size1, problem_size2,
    tensor_a, tensor_b, tensor_a2, tensor_b2, tensor_o, 
    &gemm_op1, &gemm_op2,
    &workspace1, &workspace2);

  // Result structure
  Result result;
  // Status structure
  cutlass::Status status;

  constexpr size_t kNumEventsPerIteration = 5;

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

  // synchronize all devices before beginning profiling
  MPI_Barrier(MPI_COMM_WORLD);

  //
  // Run profiling loop
  //
  for (int iter = 0; iter < options.iterations; ++iter) {
    // Record an event at the start of the GEMM
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 0], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch the first GEMM kernel on this process/GPU
    status = gemm_op1(stream);
    CUTLASS_CHECK(status);

    // Record an event when the first GEMM is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 1], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch the ReLu kernel on each device on this process/GPU
    cutlass::reference::device::TensorReLu(tensor_b2.device_view());

    // Record an event when the ReLU is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 2], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch the second GEMM kernel on each device on this process/GPU
    status = gemm_op2(stream);
    CUTLASS_CHECK(status);

    // Record an event when the second GEMM is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 3], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    
    // TODO(roshan): Replace with MSCCLPP AllReduce
    // Launch all-reduce on each device on this process/GPU
    NCCLCHECK(ncclAllReduce((const void*)tensor_o.device_data(), (void*)tensor_o.device_data(), 
          tensor_o.size(), ElementNcclAllreduce, ncclSum,
          nccl_comm, stream));

    // Record an event when the AllReduce is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 4], stream);
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

  // Measure elapsed computation and communication time
  float gemm1_time_ms{0.0f};
  float relu_time_ms{0.0f};
  float gemm2_time_ms{0.0f};
  float comm_time_ms{0.0f};
  float runtime_ms{0.0f};
  float time_ms;
  for (int iter = 0; iter < options.iterations; ++iter) {
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 0], 
      events[iter * kNumEventsPerIteration + 1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    gemm1_time_ms += time_ms;
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 1], 
      events[iter * kNumEventsPerIteration + 2]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    relu_time_ms += time_ms;
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 2], 
      events[iter * kNumEventsPerIteration + 3]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    gemm2_time_ms += time_ms;
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 3], 
      events[iter * kNumEventsPerIteration + 4]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    comm_time_ms += time_ms;
  }
  gemm1_time_ms /= static_cast<float>(options.iterations);
  relu_time_ms /= static_cast<float>(options.iterations);
  gemm2_time_ms /= static_cast<float>(options.iterations);
  comm_time_ms /= static_cast<float>(options.iterations);
  result.error = cudaEventElapsedTime(&time_ms, 
    events[0], 
    events[options.iterations * kNumEventsPerIteration - 1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }
  runtime_ms = time_ms / static_cast<float>(options.iterations);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  // TODO(roshan) skip reference check if reference_check = false
  ElementReference alpha = ElementReference(1);
  ElementReference beta = ElementReference(0);

  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementReference,
                                  LayoutInputA1,
                                  ElementReference,
                                  LayoutInputB1,
                                  ElementReference,
                                  LayoutIntermediateB2,
                                  ElementReference,
                                  ElementReference>
      gemm_device1;        
  cutlass::reference::device::Gemm<ElementReference,
                                  LayoutInputA2,
                                  ElementReference,
                                  LayoutIntermediateB2,
                                  ElementReference,
                                  LayoutOutput,
                                  ElementReference,
                                  ElementReference>
      gemm_device2;

  // Then launch device reference gemm kernel
  gemm_device1(problem_size1,
              alpha,
              tensor_ref_a.device_ref(),
              tensor_ref_b.device_ref(),
              beta,
              tensor_ref_b2.device_ref(), // does NOT matter because beta = 0
              tensor_ref_b2.device_ref());
  cutlass::reference::device::TensorReLu(tensor_ref_b2.device_view());
  gemm_device2(problem_size2,
              alpha,
              tensor_ref_a2.device_ref(),
              tensor_ref_b2.device_ref(),
              beta,
              tensor_ref_o.device_ref(), // does NOT matter because beta = 0
              tensor_ref_o.device_ref());

  // Launch all-reduce on this process/GPU
  NCCLCHECK(ncclAllReduce((const void*)tensor_ref_o.device_data(), (void*)tensor_ref_o.device_data(), 
        tensor_ref_o.size(), ElementNcclAllreduceReference, ncclSum, nccl_comm, 
        0)); // use the default stream

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_o.sync_host();
  tensor_ref_o.sync_host();

  // Then check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = relatively_equal_array(
    tensor_o.host_data(), tensor_ref_o.host_data(), tensor_o.size(), !options.csv);

  MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

  if (passed) {
    if (options.csv) {
      if (rank == 0) {
        std::cout << "Hidden size,Batch size,#Ranks,#DevicesPerNode,GEMM1 Time (ms),ReLU Time (ms),GEMM2 Time (ms),Comm. Time (ms),Total Time (ms),GFLOPS" << std::endl;
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << options.hidden_size << ",";
        os << options.batch_size << ",";
        os << num_ranks << ",";
        os << options.num_devices_per_node << ",";
        os << gemm1_time_ms << ",";
        os << relu_time_ms << ",";
        os << gemm2_time_ms << ",";
        os << comm_time_ms << ",";
        os << runtime_ms << ",";
        os << options.gflops(runtime_ms + comm_time_ms / 1000.0) << std::endl;
        std::cout << os.str();
      }
    } else {
      if (rank == 0) {
        std::cout << "Hidden size\t| Batch size\t| Rank\t| GEMM1 Time (ms)\t| ReLU Time (ms)\t| GEMM2 Time (ms)\t| Comm. Time (ms)\t| Total Time (ms)\t| GFLOPS" << std::endl;
      }
    
      MPI_Barrier(MPI_COMM_WORLD);
    
      std::ostringstream os;
      os << std::fixed << std::setprecision(3);
      os << options.hidden_size << "\t\t| ";
      os << options.batch_size << "\t\t| ";
      os << rank << "\t| ";
      os << gemm1_time_ms << "\t\t\t| ";
      os << relu_time_ms << "\t\t\t| ";
      os << gemm2_time_ms << "\t\t\t| ";
      os << comm_time_ms << "\t\t\t| ";
      os << runtime_ms << "\t\t\t| ";
      os << options.gflops(runtime_ms + comm_time_ms / 1000.0) << std::endl;
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

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}
