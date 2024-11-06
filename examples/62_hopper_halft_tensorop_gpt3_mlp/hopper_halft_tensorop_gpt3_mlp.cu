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

// #include "cutlass/gemm/kernel/tile_scheduler.hpp"
// #include "cutlass/epilogue/dispatch_policy.hpp"

#ifndef HIDDEN_SIZE
#define HIDDEN_SIZE 12288
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 1024
#endif

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

// This section selects tuned hyperparameters specific to the batch size
// that is chosen at compile-time 
// (hidden size is assumed to be 12288)
// (only selected batch sizes are supported: unsupported ones default to a batch size of 1)

#if BATCH_SIZE == 2048

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::PersistentScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 1792

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::StreamKScheduler;
using TileShape2        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 1536

// Profiler recommeneded NoSmemWarpSpecialized 
// but that leads to compilation errors due to fusion with ReLU
using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::StreamKScheduler;
using TileShape2        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 1280

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::PersistentScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 1024

// Profiler recommeneded NoSmemWarpSpecialized 
// but that leads to compilation errors due to fusion with ReLU
using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 512

// Profiler recommeneded NoSmemWarpSpecialized 
// but that leads to compilation errors due to fusion with ReLU
using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_256,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecialized;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 256

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecialized;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;  
using KernelScheduler1  = cutlass::gemm::PersistentScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::StreamKScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 128

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 64

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 32

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 16

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 8

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 4

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#elif BATCH_SIZE == 2

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#else 

#define BATCH_SIZE 1

using EpilogueSchedule1 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule1   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler1  = cutlass::gemm::StreamKScheduler;
using TileShape1        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape1     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using EpilogueSchedule2 = cutlass::epilogue::TmaWarpSpecializedCooperative;     
using KernelSchedule2   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;  
using KernelScheduler2  = cutlass::gemm::PersistentScheduler;
using TileShape2        = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape2     = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator      = float;            // <- data type of accumulator
using ElementInputA1          = cutlass::half_t;  // <- data type of elements in input matrix A
using ElementInputB1          = cutlass::half_t;  // <- data type of elements in input matrix B
using ElementInputA2          = cutlass::half_t;  // <- data type of elements in input matrix A2
using ElementIntermediateB2   = cutlass::half_t;  // <- data type of elements in intermediate matrix B2
using ElementOutput           = cutlass::half_t;  // <- data type of elements in output matrix D
constexpr ncclDataType_t ElementNcclAllreduce 
                              = ncclHalf;         // <- data type for reducing elements in output matrix D

// Use wider type for reference kernel
using ElementReference        = float;
constexpr ncclDataType_t ElementNcclAllreduceReference 
                              = ncclFloat;

// The code section below describes matrix layout of input and output matrices. 
// All matrices are in ColumnMajor
using LayoutInputA1         = cutlass::layout::ColumnMajor;
using LayoutInputB1         = cutlass::layout::ColumnMajor;
using LayoutInputA2         = cutlass::layout::ColumnMajor;
using LayoutIntermediateB2  = cutlass::layout::ColumnMajor;
using LayoutOutput          = cutlass::layout::ColumnMajor;

// Memory access granularity/alignment of matrices in units of elements (up to 16 bytes)
constexpr int AlignmentInputA1        = 128 / cutlass::sizeof_bits<ElementInputA1>::value;
constexpr int AlignmentInputB1        = 128 / cutlass::sizeof_bits<ElementInputB1>::value;
constexpr int AlignmentInputA2        = 128 / cutlass::sizeof_bits<ElementInputA2>::value;
constexpr int AlignmentIntermediateB2 = 128 / cutlass::sizeof_bits<ElementIntermediateB2>::value;
constexpr int AlignmentOutput         = 128 / cutlass::sizeof_bits<ElementOutput>::value;

// Core kernel configurations for both GEMMs
using ArchTag           = cutlass::arch::Sm90;            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass     = cutlass::arch::OpClassTensorOp; // Operator class tag

// Core kernel configurations for GEMM1
// Fuse ReLU with the epilogue
using CollectiveEpilogue1 = typename cutlass::epilogue::collective::CollectiveBuilder<
                              ArchTag, OperatorClass,
                              TileShape1, ClusterShape1,
                              cutlass::epilogue::collective::EpilogueTileAuto,
                              ElementAccumulator, ElementAccumulator,
                              ElementIntermediateB2, LayoutIntermediateB2, AlignmentIntermediateB2,
                              ElementIntermediateB2, LayoutIntermediateB2, AlignmentIntermediateB2,
                              EpilogueSchedule1,
                              cutlass::epilogue::fusion::LinCombEltAct<cutlass::epilogue::thread::ReLu, ElementIntermediateB2, ElementAccumulator>
                            >::CollectiveOp;
using StageCountType1     = cutlass::gemm::collective::StageCountAutoCarveout<
                              static_cast<int>(sizeof(typename CollectiveEpilogue1::SharedStorage))>;
using CollectiveMainloop1 = typename cutlass::gemm::collective::CollectiveBuilder<
                              ArchTag, OperatorClass,
                              ElementInputA1, LayoutInputA1, AlignmentInputA1,
                              ElementInputB1, LayoutInputB1, AlignmentInputB1,
                              ElementAccumulator,
                              TileShape1, ClusterShape1,
                              StageCountType1,
                              KernelSchedule1
                            >::CollectiveOp;
using GemmKernel1         = cutlass::gemm::kernel::GemmUniversal<
                              Shape<int,int,int>, // Indicates ProblemShape
                              CollectiveMainloop1,
                              CollectiveEpilogue1,
                              KernelScheduler1
                            >;
using Gemm1               = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1>;
static_assert(std::is_same<ElementInputA1, typename Gemm1::ElementA>::value);
static_assert(std::is_same<ElementInputB1, typename Gemm1::ElementB>::value);
static_assert(std::is_same<ElementIntermediateB2, typename Gemm1::ElementC>::value);
static_assert(std::is_same<ElementIntermediateB2, typename Gemm1::ElementD>::value);

// Core kernel configurations for GEMM2
using CollectiveEpilogue2 = typename cutlass::epilogue::collective::CollectiveBuilder<
                              ArchTag, OperatorClass,
                              TileShape2, ClusterShape2,
                              cutlass::epilogue::collective::EpilogueTileAuto,
                              ElementAccumulator, ElementAccumulator,
                              ElementOutput, LayoutOutput, AlignmentOutput,
                              ElementOutput, LayoutOutput, AlignmentOutput,
                              EpilogueSchedule2
                            >::CollectiveOp;
using StageCountType2     = cutlass::gemm::collective::StageCountAutoCarveout<
                              static_cast<int>(sizeof(typename CollectiveEpilogue2::SharedStorage))>;
using CollectiveMainloop2 = typename cutlass::gemm::collective::CollectiveBuilder<
                              ArchTag, OperatorClass,
                              ElementInputA2, LayoutInputA2, AlignmentInputA2,
                              ElementIntermediateB2, LayoutIntermediateB2, AlignmentIntermediateB2,
                              ElementAccumulator,
                              TileShape2, ClusterShape2,
                              StageCountType2,
                              KernelSchedule2
                            >::CollectiveOp;
using GemmKernel2         = cutlass::gemm::kernel::GemmUniversal<
                              Shape<int,int,int>, // Indicates ProblemShape
                              CollectiveMainloop2,
                              CollectiveEpilogue2,
                              KernelScheduler2
                            >;
using Gemm2               = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2>;
static_assert(std::is_same<ElementInputA2, typename Gemm2::ElementA>::value);
static_assert(std::is_same<ElementIntermediateB2, typename Gemm2::ElementB>::value);
static_assert(std::is_same<ElementOutput, typename Gemm2::ElementC>::value);
static_assert(std::is_same<ElementOutput, typename Gemm2::ElementD>::value);

using StrideInputA1         = typename Gemm1::GemmKernel::StrideA;
using StrideInputB1         = typename Gemm1::GemmKernel::StrideB;
using StrideIntermediateD1  = typename Gemm1::GemmKernel::StrideD;
using StrideInputA2         = typename Gemm2::GemmKernel::StrideA;
using StrideIntermediateB2  = typename Gemm2::GemmKernel::StrideB;
using StrideOutput          = typename Gemm2::GemmKernel::StrideD;
using Arguments1            = typename Gemm1::Arguments;
using Arguments2            = typename Gemm2::Arguments;

// Reference device GEMM implementation type
using DeviceGemmReference1  = cutlass::reference::device::Gemm<
                                ElementReference,
                                LayoutInputA1,
                                ElementReference,
                                LayoutInputB1,
                                ElementReference,
                                LayoutIntermediateB2,
                                ElementAccumulator,
                                ElementAccumulator>;
using DeviceGemmReference2  = cutlass::reference::device::Gemm<
                                ElementReference,
                                LayoutInputA2,
                                ElementReference,
                                LayoutIntermediateB2,
                                ElementReference,
                                LayoutOutput,
                                ElementAccumulator,
                                ElementAccumulator>;

// alpha and beta for dot product computation
constexpr uint32_t kAlpha{1};
constexpr uint32_t kBeta{0};

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
  bool use_mscclpp_smcopy;
  bool use_mscclpp_nvls;
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

  cutlass::gemm::GemmCoord problem_size1;
  cutlass::gemm::GemmCoord problem_size2;

  Options():
    help(false),
    csv(false),
    hidden_size(HIDDEN_SIZE),
    batch_size(BATCH_SIZE),
    reference_check(true),
    use_nccl(false),
    use_mscclpp_smcopy(false),
    use_mscclpp_nvls(false),
    inplace_mscclpp(true), 
    iterations(20),
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

  // void createSMChannels () {
  //   // Initialize communicator and create one connection per peer
  //   mscclpp::Communicator comm(bootstrap);
  //   std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
  //   std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  //   for (int i = 0; i < nRanks; i++) {
  //     if (i == rank) continue;
  //     connectionFutures.push_back(comm.connectOnSetup(i, 0, mscclpp::Transport::CudaIpc));
  //   }
  //   comm.setup();
  //   std::transform(connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
  //                 [](const auto& future) { return future.get(); });

  //   // Create one semaphore per connection per SM
  //   int nSMs = 132;
  //   std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  //   for (int i = 0; i < nSMs; i++) {
  //     for (auto &conn : connections) {
  //       smSemaphores.emplace_back(
  //           std::make_shared<mscclpp::SmDevice2DeviceSemaphore>(comm, conn));
  //     }
  //   }
  //   comm.setup();
  // }

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
    size_t output_buffer_size = problem_size2.m() * problem_size2.n() * sizeof(ElementOutput);

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

    if (cmd.check_cmd_line_flag("do_not_verify")) {
      reference_check = false;
    }

    if (cmd.check_cmd_line_flag("use_nccl")) {
      use_nccl = true;
    } else if (cmd.check_cmd_line_flag("use_mscclpp_smcopy")) {
      use_mscclpp_smcopy = true;
    } else if (cmd.check_cmd_line_flag("use_mscclpp_nvls")) {
      use_mscclpp_nvls = true;
      if (cmd.check_cmd_line_flag("use_multicast_reduce")) {
        inplace_mscclpp = false;
      } else {
        inplace_mscclpp = true;
      }
    }

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("comm_block_size", mscclpp_block_size);
    cmd.get_cmd_line_argument("comm_grid_size", mscclpp_grid_size);
    assert(mscclpp_block_size >= num_ranks);

    if (use_mscclpp_nvls || use_mscclpp_smcopy) {
      // Create MSCCL peer to peer connections
      createP2PConnections();
      
      if (use_mscclpp_nvls)
        // Now create NVLS Connection: requires output buffer size, which depends on problem sizes
        createNvlsConnection();
    }
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

    printf("[%d,%d] x [%d,%d] HALFT tensor op Matrix Multiply fused with ReLU\n", \
      problem_size1.m(), problem_size1.k(), 
      problem_size1.k(), problem_size1.n());

    printf("[%d,%d] x [%d,%d] HALFT tensor op Matrix Multiply\n", \
      problem_size2.m(), problem_size2.k(), 
      problem_size2.k(), problem_size2.n());

    printf("[%d,%d] HALFT tensor op AllReduce\n", \
      problem_size2.m(),  problem_size2.n());

    fflush(stdout);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "61_ampere_halft_tensorop_gpt3_mlp example\n\n"
      << "  This example uses the CUTLASS Library to execute halft tensorop GPT3 MLP computation.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --csv                       If specified, prints in CSV format.\n\n"
      << "  --do_not_verify             If specified, skips verification of results using a reference implemenation.\n\n"
      << "  --use_nccl                  If specified, uses NCCL AllReduce instead of MSCCLPP implementation.\n\n"
      << "  --use_multicast_reduce      If specified, uses multicast reduce-based AllReduce with separate output memory.\n\n"
      << "  --comm_block_size           Number of threads per block to use in MSCCLPP AllReduce (default 1024).\n\n"
      << "  --comm_grid_size            Number of blocks per grid to use in MSCCLPP AllReduce (default 8).\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (default 20).\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/62_hopper_halft_tensorop_gpt3/62_hopper_halft_tensorop_2mlp_gpt3_mlp_h<hidden_size>_b<batch_size>\n\n";

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
  int block_size = 0) {
  
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

  BlockCopyKernel<WiderElement, Element><<< grid, block >>>(
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

void InitializeStrides(cutlass::gemm::GemmCoord& problem_size1, 
  cutlass::gemm::GemmCoord& problem_size2, 
  StrideInputA1* stride_a1,
  StrideInputB1* stride_b1,
  StrideIntermediateD1* stride_d1,
  StrideInputA2* stride_a2,
  StrideIntermediateB2* stride_b2,
  StrideOutput* stride_o) {

  assert(problem_size1.m() == problem_size2.k());
  assert(problem_size1.n() == problem_size2.n());

  *stride_a1  = cutlass::make_cute_packed_stride(StrideInputA1{}, 
                  cute::make_shape(problem_size1.m(), problem_size1.k(), Int<1>{}));
  *stride_b1  = cutlass::make_cute_packed_stride(StrideInputB1{}, 
                  cute::make_shape(problem_size1.n(), problem_size1.k(), Int<1>{}));
  *stride_d1  = cutlass::make_cute_packed_stride(StrideIntermediateD1{}, 
                  cute::make_shape(problem_size1.m(), problem_size1.n(), Int<1>{}));
  *stride_a2  = cutlass::make_cute_packed_stride(StrideInputA2{}, 
                  cute::make_shape(problem_size2.m(), problem_size2.k(), Int<1>{}));
  *stride_b2  = cutlass::make_cute_packed_stride(StrideIntermediateB2{}, 
                  cute::make_shape(problem_size2.n(), problem_size2.k(), Int<1>{}));
  *stride_o   = cutlass::make_cute_packed_stride(StrideOutput{}, 
                  cute::make_shape(problem_size2.m(), problem_size2.n(), Int<1>{}));
}

void InitializeMatrices(cutlass::gemm::GemmCoord& problem_size1, 
  cutlass::gemm::GemmCoord& problem_size2, 
  cutlass::DeviceAllocation<ElementInputA1>* block_a1,
  cutlass::DeviceAllocation<ElementInputB1>* block_b1,
  cutlass::DeviceAllocation<ElementInputA2>* block_a2,
  cutlass::DeviceAllocation<ElementIntermediateB2>* block_b2,
  cutlass::DeviceAllocation<ElementOutput>* block_partial_o,
  cutlass::DeviceAllocation<ElementOutput>* block_o,
  cutlass::DeviceAllocation<ElementReference>* block_ref_a1,
  cutlass::DeviceAllocation<ElementReference>* block_ref_b1,
  cutlass::DeviceAllocation<ElementReference>* block_ref_a2,
  cutlass::DeviceAllocation<ElementReference>* block_ref_b2,
  cutlass::DeviceAllocation<ElementReference>* block_ref_o, 
  ncclComm_t& nccl_comm) {

  assert(problem_size1.m() == problem_size2.k());
  assert(problem_size1.n() == problem_size2.n());

  block_a1->reset(problem_size1.m() * problem_size1.k());
  block_b1->reset(problem_size1.k() * problem_size1.n());
  block_a2->reset(problem_size2.m() * problem_size2.k());
  block_b2->reset(problem_size2.k() * problem_size2.n());
  block_partial_o->reset(problem_size2.m() * problem_size2.n());
  block_o->reset(problem_size2.m() * problem_size2.n());
  
  block_ref_a1->reset(problem_size1.m() * problem_size1.k());
  block_ref_b1->reset(problem_size1.k() * problem_size1.n());
  block_ref_a2->reset(problem_size2.m() * problem_size2.k());
  block_ref_b2->reset(problem_size2.k() * problem_size2.n());
  block_ref_o->reset(problem_size2.m() * problem_size2.n());

  // Fill input matrices with uniform-distribution random data
  // For half_t tensors used in CUTLASS kernel, these ranges and precision for random values yield:
  // (1) 0.001f relatively equal results for hidden_size<=12k and batch_size<=2k
  // (2) 0.0001f relatively equal results for hidden_size<=1k and batch_size<=128
  cutlass::reference::device::BlockFillRandomUniform(
    block_a1->get(), 
    block_a1->size(), 
    2024, // seed
    ElementInputA1{0.3}, 
    ElementInputA1{-0.3}, 
    2);
  cutlass::reference::device::BlockFillRandomUniform(
    block_b1->get(), 
    block_b1->size(), 
    2023, // seed
    ElementInputB1{0.3}, 
    ElementInputB1{-0.3}, 
    2);
  cutlass::reference::device::BlockFillRandomUniform(
    block_a2->get(), 
    block_a2->size(), 
    2022, // seed
    ElementInputA2{0.3}, 
    ElementInputA2{-0.3}, 
    2);

  // Fill intermediate and output matrices with zeros
  typename ZeroFunc<ElementIntermediateB2>::Params params_b2;
  cutlass::reference::device::BlockForEach<ElementIntermediateB2, ZeroFunc<ElementIntermediateB2>>(
    block_b2->get(), 
    block_b2->size(), 
    params_b2);
  typename ZeroFunc<ElementOutput>::Params params_o;
  cutlass::reference::device::BlockForEach<ElementOutput, ZeroFunc<ElementOutput>>(
    block_partial_o->get(), 
    block_partial_o->size(), 
    params_o);
  cutlass::reference::device::BlockForEach<ElementOutput, ZeroFunc<ElementOutput>>(
    block_o->get(), 
    block_o->size(), 
    params_o);

  // Copy randomly generated values from input matrices to inputs for the reference kernel
  // The corresponding types of the tensors in the two kernels might be different
  assert(block_a1->size() == block_ref_a1->size());
  BlockCopy(block_ref_a1->get(), block_a1->get(), block_a1->size());
  assert(block_b1->size() == block_ref_b1->size());
  BlockCopy(block_ref_b1->get(), block_b1->get(), block_b1->size());
  assert(block_a2->size() == block_ref_a2->size());
  BlockCopy(block_ref_a2->get(), block_a2->get(), block_a2->size());

  // Fill intermediate and output matrices of reference kernel with zeros
  typename ZeroFunc<ElementReference>::Params params_ref;
  cutlass::reference::device::BlockForEach<ElementReference, ZeroFunc<ElementReference>>(
    block_ref_b2->get(), 
    block_ref_b2->size(), 
    params_ref);
  cutlass::reference::device::BlockForEach<ElementReference, ZeroFunc<ElementReference>>(
    block_ref_o->get(),
    block_ref_o->size(), 
    params_ref);

  NCCLCHECK(ncclAllReduce((const void*)block_ref_o->get(), (void*)block_ref_o->get(), 
    block_ref_o->size(), ElementNcclAllreduceReference, ncclSum,
    nccl_comm, 0));
}

Arguments1 ConstructGemm1(cutlass::gemm::GemmCoord& problem_size1, 
  cutlass::DeviceAllocation<ElementInputA1>& block_a1,
  cutlass::DeviceAllocation<ElementInputB1>& block_b1,
  cutlass::DeviceAllocation<ElementIntermediateB2>& block_b2,
  StrideInputA1& stride_a1,
  StrideInputB1& stride_b1,
  StrideIntermediateD1& stride_d1,
  Gemm1* gemm_op1,
  cutlass::device_memory::allocation<uint8_t>* workspace1) {

  ElementAccumulator alpha = ElementAccumulator{kAlpha};
  ElementAccumulator beta = ElementAccumulator{kBeta};
        
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  Arguments1 arguments1{cutlass::gemm::GemmUniversalMode::kGemm,
                          {problem_size1.m(), problem_size1.n(), problem_size1.k()},
                          {block_a1.get(), stride_a1,     // <- reference to matrix A on device 
                            block_b1.get(), stride_b1},   // <- reference to matrix B on device
                          {{alpha, beta},                 // <- tuple of alpha and beta
                            block_b2.get(), stride_d1,    // <- matric C: does NOT matter because beta = 0
                            block_b2.get(), stride_d1}};  // <- reference to matrix D on device

  cutlass::Status status;

  // Check the problem size is supported or not 
  status = gemm_op1->can_implement(arguments1);
  CUTLASS_CHECK(status);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size1 = Gemm1::get_workspace_size(arguments1);

  // Allocate workspace memory
  workspace1->reset(workspace_size1);

  return arguments1;
}

Arguments2 ConstructGemm2(cutlass::gemm::GemmCoord& problem_size2, 
  cutlass::DeviceAllocation<ElementInputA2>& block_a2,
  cutlass::DeviceAllocation<ElementIntermediateB2>& block_b2,
  cutlass::DeviceAllocation<ElementOutput>& block_o,
  StrideInputA2& stride_a2,
  StrideIntermediateB2& stride_b2,
  StrideOutput& stride_o,
  Gemm2* gemm_op2, 
  cutlass::device_memory::allocation<uint8_t>* workspace2) {

  ElementAccumulator alpha = ElementAccumulator{kAlpha};
  ElementAccumulator beta = ElementAccumulator{kBeta};
        
  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  Arguments2 arguments2{cutlass::gemm::GemmUniversalMode::kGemm,
                          {problem_size2.m(), problem_size2.n(), problem_size2.k()},
                          {block_a2.get(), stride_a2,     // <- reference to matrix A on device 
                            block_b2.get(), stride_b2},   // <- reference to matrix B on device
                          {{alpha, beta},                 // <- tuple of alpha and beta
                            block_o.get(), stride_o,      // <- matric C: does NOT matter because beta = 0
                            block_o.get(), stride_o}};    // <- reference to matrix D on device

  cutlass::Status status;

  // Check the problem size is supported or not 
  status = gemm_op2->can_implement(arguments2);
  CUTLASS_CHECK(status);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size2 = Gemm2::get_workspace_size(arguments2);

  // Allocate workspace memory
  workspace2->reset(workspace_size2);

  return arguments2;
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

  //
  // Data members
  //
  StrideInputA1 stride_a1;
  StrideInputB1 stride_b1;
  StrideIntermediateD1 stride_d1;
  StrideInputA2 stride_a2;
  StrideIntermediateB2 stride_b2;
  StrideOutput stride_o;
  // Create matrices on this process/GPU
  cutlass::DeviceAllocation<ElementInputA1> block_a1;
  cutlass::DeviceAllocation<ElementInputB1> block_b1;
  cutlass::DeviceAllocation<ElementInputA2> block_a2;
  cutlass::DeviceAllocation<ElementIntermediateB2> block_b2;
  cutlass::DeviceAllocation<ElementOutput> block_partial_o;
  cutlass::DeviceAllocation<ElementOutput> block_o;
  // Create matrices on this process/GPU for the reference kernel
  cutlass::DeviceAllocation<ElementReference> block_ref_a1;
  cutlass::DeviceAllocation<ElementReference> block_ref_b1;
  cutlass::DeviceAllocation<ElementReference> block_ref_a2;
  cutlass::DeviceAllocation<ElementReference> block_ref_b2;
  cutlass::DeviceAllocation<ElementReference> block_ref_o;

  // Initialize strides on this process/GPU
  InitializeStrides(problem_size1, problem_size2, 
    &stride_a1, &stride_b1, &stride_d1, &stride_a2, &stride_b2, &stride_o);

  // Initialize matrices on this process/GPU
  InitializeMatrices(problem_size1, problem_size2,
    &block_a1, &block_b1, &block_a2, &block_b2, &block_partial_o, &block_o,
    &block_ref_a1, &block_ref_b1, &block_ref_a2, &block_ref_b2, &block_ref_o, 
    nccl_comm);

  // Create the multicast pointer for the output tensor
  ElementOutput* block_o_mc_ptr = nullptr;
  if (options.use_mscclpp_nvls) {
    std::shared_ptr<char> block_o_mc = options.nvls_connection->bindAllocatedCuda(block_o.getHandle(), block_o.bytes_allocated());
    block_o_mc_ptr = (ElementOutput*)block_o_mc.get();
  }

  // Create the GEMM ops
  Gemm1 gemm_op1;
  Gemm2 gemm_op2;
  // Create the workspaces for the GEMM ops
  cutlass::device_memory::allocation<uint8_t> workspace1;
  cutlass::device_memory::allocation<uint8_t> workspace2;

  // Construct the GEMM ops on this process/GPU
  Arguments1 arguments1 = ConstructGemm1(problem_size1, 
    block_a1, block_b1, 
    block_b2, 
    stride_a1, stride_b1, stride_d1, 
    &gemm_op1, &workspace1);
  Arguments2 arguments2 = ConstructGemm2(problem_size2,
    // NCCL and inplace MSCCLPP do not use a separate partial output tensor
    block_a2, block_b2, 
    (options.use_nccl || options.inplace_mscclpp) ? block_o : block_partial_o,
    stride_a2, stride_b2, stride_o,
    &gemm_op2, &workspace2);

  // Result structure
  Result result;
  // Status structure
  cutlass::Status status;

  constexpr size_t kNumEventsPerIteration = 4;

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
  printf("1238\n");
  // synchronize all devices before beginning profiling
  MPI_Barrier(MPI_COMM_WORLD);

  //
  // Run profiling loop
  //
  for (int iter = 0; iter < options.iterations; ++iter) {
    // Initialize CUTLASS kernels with arguments and workspace pointer
    status = gemm_op1.initialize(arguments1, workspace1.get());
    CUTLASS_CHECK(status);
    status = gemm_op2.initialize(arguments2, workspace2.get());
    CUTLASS_CHECK(status);

    if (!options.use_nccl && !options.inplace_mscclpp) {
      // Zero out the output tensor because partial outputs from each device are reduced onto it
      typename ZeroFunc<ElementOutput>::Params params_o;
      cutlass::reference::device::BlockForEach<ElementOutput, ZeroFunc<ElementOutput>>(
        block_o.get(), 
        block_o.size(), 
        params_o, 
        0, 0, stream);
    }

    // Record an event at the start of the GEMM
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 0], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch the first GEMM kernel on this process/GPU
    status = gemm_op1.run(stream);
    CUTLASS_CHECK(status);

    // Record an event when the first GEMM is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 1], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }

    // Launch the second GEMM kernel on each device on this process/GPU
    status = gemm_op2.run(stream);
    CUTLASS_CHECK(status);

    // Record an event when the second GEMM is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 2], stream);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    // Launch all-reduce on each device on this process/GPU
    if (options.use_nccl) {
      NCCLCHECK(ncclAllReduce((const void*)block_o.get(), (void*)block_o.get(), 
            block_o.size(), ElementNcclAllreduce, ncclSum,
            nccl_comm, stream));
    } else if (options.use_mscclpp_smcopy) {
      // int num_threads_per_block = options.mscclpp_block_size;
      // int num_blocks = options.mscclpp_grid_size;
      // mscclppSMCopyAllReduceInplaceSum<<<num_blocks, num_threads_per_block, 0, stream>>>(block_o_mc_ptr, block_o.size(),
                                                                                  // rank, num_ranks);
    } else if (options.use_mscclpp_nvls) {
      if (options.inplace_mscclpp) {
        int num_threads_per_block = options.mscclpp_block_size;
        int num_blocks = options.mscclpp_grid_size;
        mscclppNVLSAllReduceInplaceSum<<<num_blocks, num_threads_per_block, 0, stream>>>(block_o_mc_ptr, block_o.size(),
                                                                                    rank, num_ranks);
      } else {
        int num_threads_per_block = options.mscclpp_block_size;
        int num_blocks = options.mscclpp_grid_size;
        assert(block_partial_o.size() == block_o.size());
        mscclppNVLSAllReduceSum<<<num_blocks, num_threads_per_block, 0, stream>>>(block_partial_o.get(), block_o_mc_ptr, block_o.size(),
                                                                              rank, num_ranks);
      }
    }
    // Record an event when the AllReduce is complete
    result.error = cudaEventRecord(events[iter * kNumEventsPerIteration + 3], stream);
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
    gemm2_time_ms += time_ms;
    result.error = cudaEventElapsedTime(&time_ms, 
      events[iter * kNumEventsPerIteration + 2], 
      events[iter * kNumEventsPerIteration + 3]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
    comm_time_ms += time_ms;
  }
  gemm1_time_ms /= static_cast<float>(options.iterations);
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
  // TODO(roshan) do this before running multiple iterations
  // Create instantiation for device reference gemm kernel
  DeviceGemmReference1 gemm_device1;        
  DeviceGemmReference2 gemm_device2;

  cutlass::TensorRef tensor_ref_a1(block_ref_a1.get(), 
                                    Gemm1::LayoutA::packed({problem_size1.m(), problem_size1.k()}));
  cutlass::TensorRef tensor_ref_b1(block_ref_b1.get(),
                                    Gemm1::LayoutB::packed({problem_size1.k(), problem_size1.n()}));
  cutlass::TensorRef tensor_ref_a2(block_ref_a2.get(),
                                    Gemm2::LayoutA::packed({problem_size2.m(), problem_size2.k()}));
  cutlass::TensorView tensor_ref_b2(block_ref_b2.get(),
                                      Gemm2::LayoutB::packed({problem_size2.k(), problem_size2.n()}),
                                      {problem_size2.k(), problem_size2.n()});
  cutlass::TensorRef tensor_ref_o(block_ref_o.get(),
                                    Gemm2::LayoutD::packed({problem_size2.m(), problem_size2.n()}));

  ElementAccumulator alpha = ElementAccumulator{kAlpha};
  ElementAccumulator beta = ElementAccumulator{kBeta};

  // Then launch device reference gemm kernel
  gemm_device1(problem_size1,
                alpha,
                tensor_ref_a1,
                tensor_ref_b1,
                beta,
                tensor_ref_b2,  // does NOT matter because beta = 0
                tensor_ref_b2);
  cutlass::reference::device::TensorReLu(tensor_ref_b2);
  gemm_device2(problem_size2,
                alpha,
                tensor_ref_a2,
                tensor_ref_b2,
                beta,
                tensor_ref_o,   // does NOT matter because beta = 0
                tensor_ref_o);  

  // Launch all-reduce on this process/GPU
  NCCLCHECK(ncclAllReduce((const void*)block_ref_o.get(), (void*)block_ref_o.get(), 
        block_ref_o.size(), ElementNcclAllreduceReference, ncclSum, nccl_comm, 
        0)); // use the default stream

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Then check if output from CUTLASS kernel and reference kernel are equal or not
  assert(block_o.size() == block_ref_o.size());
  // This is the maximum relative error we expect from the relative equality check
  // A higher value can lead to errors due to quantized/lower-precision values 
  // that are used in the CUTLASS kernel
  ElementReference epsilon(0.001f);
  ElementReference nonzero_floor(std::numeric_limits<ElementReference>::min());
  bool passed = BlockCompareRelativelyEqual(
    block_ref_o.get(), block_o.get(), block_o.size(), epsilon, nonzero_floor);

  MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);

  if (passed) {
    if (options.csv) {
      if (rank == 0) {
        std::cout << "Hidden size,Batch size,#Ranks,#DevicesPerNode,GEMM1 Time (ms),GEMM2 Time (ms),Comm. Time (ms),Total Time (ms),GFLOPS" << std::endl;
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << options.hidden_size << ",";
        os << options.batch_size << ",";
        os << num_ranks << ",";
        os << options.num_devices_per_node << ",";
        os << gemm1_time_ms << ",";
        os << gemm2_time_ms << ",";
        os << comm_time_ms << ",";
        os << runtime_ms << ",";
        os << options.gflops(runtime_ms / 1000.0) << std::endl;
        std::cout << os.str();
      }
    } else {
      if (rank == 0) {
        std::cout << "Hidden size\t| Batch size\t| Rank\t| GEMM1 Time (ms)\t| GEMM2 Time (ms)\t| Comm. Time (ms)\t| Total Time (ms)\t| GFLOPS" << std::endl;
      }
    
      MPI_Barrier(MPI_COMM_WORLD);
    
      std::ostringstream os;
      os << std::fixed << std::setprecision(3);
      os << options.hidden_size << "\t\t| ";
      os << options.batch_size << "\t\t| ";
      os << rank << "\t| ";
      os << gemm1_time_ms << "\t\t\t| ";
      os << gemm2_time_ms << "\t\t\t| ";
      os << comm_time_ms << "\t\t\t| ";
      os << runtime_ms << "\t\t\t| ";
      os << options.gflops(runtime_ms / 1000.0) << std::endl;
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
  
  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
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

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  return run(options);
#else
  return 0;
#endif
}
