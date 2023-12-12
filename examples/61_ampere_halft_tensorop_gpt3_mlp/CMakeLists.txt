
# Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(MPI REQUIRED)

function(cutlass_mpi_example_add_executable NAME HIDDEN_SIZE BATCH_SIZE)

  set(options)
  set(oneValueArgs DISABLE_TESTS)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(FULL_NAME ${NAME}_h${HIDDEN_SIZE}_b${BATCH_SIZE})

  cutlass_add_executable(${FULL_NAME} ${__UNPARSED_ARGUMENTS})

  target_compile_definitions(${FULL_NAME} PRIVATE HIDDEN_SIZE=${HIDDEN_SIZE} BATCH_SIZE=${BATCH_SIZE})

  add_dependencies(cutlass_examples ${FULL_NAME})

  target_link_libraries(
    ${FULL_NAME}
    PRIVATE
    CUTLASS
    cutlass_tools_util_includes
    $<$<BOOL:${CUTLASS_ENABLE_CUBLAS}>:nvidia::cublas>
    cuda
    nccl
    MPI::MPI_CXX
    )

  target_include_directories(
    ${FULL_NAME}
    PRIVATE
    ${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CUTLASS_EXAMPLES_UTILS_DIR}
    )

  install(
    TARGETS ${FULL_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

  cutlass_add_executable_tests(
    test_examples_${FULL_NAME} ${FULL_NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_examples ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    DISABLE_TESTS ${__DISABLE_TESTS}
    )

endfunction()

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  2048
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  1792
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  1536
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  1280
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  1024
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  512
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  256
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  128
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  64
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  32
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  16
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  8
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  4
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  2
  ampere_halft_tensorop_gpt3_mlp.cu
  )

cutlass_mpi_example_add_executable(
  61_ampere_halft_tensorop_gpt3_mlp
  12288
  1
  ampere_halft_tensorop_gpt3_mlp.cu
  )
