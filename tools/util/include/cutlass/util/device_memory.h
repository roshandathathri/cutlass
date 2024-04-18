/******************************************************************************
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
 ******************************************************************************/

#pragma once

/**
 * \file
 * \brief C++ interface to CUDA device memory management functions.
 */

#include <memory>

#include <cuda.h>

#include "cutlass/platform/platform.h"
#include "cutlass/numeric_types.h"
#include "exceptions.h"

namespace cutlass {
namespace device_memory {

/******************************************************************************
 * Allocation lifetime
 ******************************************************************************/
struct Memory {
  /// Number of bytes on the current CUDA device
  size_t bytes;

  /// Pointer to the memory on the current CUDA device
  CUdeviceptr ptr;

  /// Handle to the memory on the current CUDA device
  CUmemGenericAllocationHandle handle;

  Memory() : bytes(0), ptr(0), handle(0) {}

  Memory(size_t _bytes) : bytes(_bytes), ptr(0), handle(0) {
    if (bytes == 0) {
      return;
    }

    int deviceId = -1;
    cudaError_t cuda_error = cudaGetDevice(&deviceId);
    if (cuda_error != cudaSuccess) {
      std::cout << "Failed to get device" << std::endl;
      throw cuda_exception("Failed to get device", cuda_error);
    }

    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = deviceId;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity = 0;
    CUresult cu_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (cu_result != CUDA_SUCCESS) {
      std::cout << "Failed to get the minimum mem allocation granularity" << std::endl;
      throw cuda_exception("Failed to get the minimum mem allocation granularity", cuda_error);
    }

    // Pad the allocation size to a multiple of the granularity
    bytes = ((bytes + granularity - 1) / granularity) * granularity;

    // create a memory handle
    cu_result = cuMemCreate(&handle, bytes, &prop, 0);
    if (cu_result != CUDA_SUCCESS) {
      std::cout << "Failed to create a memory handle for " << bytes << " bytes : " << cu_result << std::endl;
      throw cuda_exception("Failed to create a memory handle", cuda_error);
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = deviceId;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Reserve virtual address space
    cu_result = cuMemAddressReserve(&ptr, bytes, granularity, 0U, 0);
    if (cu_result != CUDA_SUCCESS) {
      std::cout << "Failed to reserve virtual address space" << std::endl;
      throw cuda_exception("Failed to reserve virtual address space", cuda_error);
    }

    // Map the memory handle to the reserved virtual address space
    cu_result = cuMemMap(ptr, bytes, 0, handle, 0);
    if (cu_result != CUDA_SUCCESS) {
      std::cout << "Failed to map the handle to the virtual address space" << std::endl;
      throw cuda_exception("Failed to map the handle to the virtual address space", cuda_error);
    }

    // Set the access flags for the virtual address space
    cu_result = cuMemSetAccess(ptr, bytes, &accessDesc, 1);
    if (cu_result != CUDA_SUCCESS) {
      std::cout << "Failed to set access for the allocated memory" << std::endl;
      throw cuda_exception("Failed to set access for the allocated memory", cuda_error);
    }

    // Initialize the memory to zero
    // cuda_error = cudaMemset(ptr, 0, bytes);
  }

  ~Memory() {
    if (ptr) {
      CUresult cu_result = cuMemUnmap(ptr, bytes);
      if (cu_result != CUDA_SUCCESS) {
        std::cout << "Failed to unmap device memory" << std::endl;
        // noexcept
        //                throw cuda_exception("cuMemUnmap() failed", cuda_error);
        return;
      }
      cu_result = cuMemAddressFree(ptr, bytes);
      if (cu_result != CUDA_SUCCESS) {
        std::cout << "Failed to free device virtual memory" << std::endl;
        // noexcept
        //                throw cuda_exception("cuMemAddressFree() failed", cuda_error);
        return;
      }
      cu_result = cuMemRelease(handle);
      if (cu_result != CUDA_SUCCESS) {
        std::cout << "Failed to release memory handle" << std::endl;
        // noexcept
        //            throw cuda_exception("Failed to release memory handle", cuda_error);
        return;
      }
    }
  }
};

/******************************************************************************
 * Data movement
 ******************************************************************************/

template <typename T>
void copy(T* dst, T const* src, size_t count, cudaMemcpyKind kind) {
  // TODO: should use bytes() in DeviceAllocation
  size_t bytes = count * sizeof_bits<T>::value / 8;
  if (bytes == 0 && count > 0)
    bytes = 1;
  cudaError_t cuda_error = (cudaMemcpy(dst, src, bytes, kind));
  if (cuda_error != cudaSuccess) {
    throw cuda_exception("cudaMemcpy() failed", cuda_error);
  }
}

template <typename T>
void copy_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToHost);
}

template <typename T>
void copy_device_to_device(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyDeviceToDevice);
}

template <typename T>
void copy_host_to_host(T* dst, T const* src, size_t count = 1) {
  copy(dst, src, count, cudaMemcpyHostToHost);
}

/// Copies elements from device memory to host-side range
template <typename OutputIterator, typename T>
void insert_to_host(OutputIterator begin, OutputIterator end, T const* device_begin) {
  size_t elements = end - begin;
  copy_to_host(&*begin, device_begin, elements);
}

/// Copies elements to device memory from host-side range
template <typename T, typename InputIterator>
void insert_to_device(T* device_begin, InputIterator begin, InputIterator end) {
  size_t elements = end - begin;
  copy_to_device(device_begin, &*begin, elements);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace device_memory

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DeviceAllocation {
public:
  //
  // Data members
  //

  /// Number of elements of T allocated on the current CUDA device
  size_t capacity;

  /// Smart pointer managing the allocation
  platform::unique_ptr<device_memory::Memory> smart_ptr;

public:

  //
  // Static methods
  //

  /// Static member to compute the number of bytes needed for a given number of elements
  static size_t bytes(size_t elements) {
    if constexpr (sizeof_bits<T>::value < 8) {
      size_t constexpr kElementsPerByte = 8 / sizeof_bits<T>::value;
      return elements / kElementsPerByte;
    }
    else {
      size_t constexpr kBytesPerElement = sizeof_bits<T>::value / 8;
      return elements * kBytesPerElement;
    }
  }

public:

  //
  // Methods
  //

  /// Constructor: allocates no memory
  DeviceAllocation() : capacity(0) {}

  /// Constructor: allocates \p capacity elements on the current CUDA device
  DeviceAllocation(size_t _capacity) : 
    capacity(_capacity), smart_ptr(new device_memory::Memory(bytes(_capacity))) {}

  /// Copy constructor
  DeviceAllocation(DeviceAllocation const &p): 
    capacity(p.capacity), smart_ptr(new device_memory::Memory(bytes(p.capacity))) {

    device_memory::copy_device_to_device(get(), p.get(), capacity);
  }

  /// Move constructor
  DeviceAllocation(DeviceAllocation &&p) {
    std::swap(capacity, p.capacity);
    std::swap(smart_ptr, p.smart_ptr);
  }

  /// Destructor
  ~DeviceAllocation() { reset(); }

  /// Returns a pointer to the managed allocation
  T* get() const { return (T*)smart_ptr.get()->ptr; }

  /// Returns a handle to the managed allocation
  CUmemGenericAllocationHandle getHandle() const { return smart_ptr.get()->handle; }

  /// Deletes the managed object and resets capacity to zero
  void reset() {
    capacity = 0;
    smart_ptr.reset();
  }

  /// Deletes managed object, if owned, and allocates a new object
  void reset(size_t _capacity) {
    capacity = _capacity;
    smart_ptr.reset(new device_memory::Memory(bytes(_capacity)));
  }

  /// Allocates a new buffer and copies the old buffer into it. The old buffer is then released.
  void reallocate(size_t new_capacity) {
    platform::unique_ptr<device_memory::Memory> new_smart_ptr(new device_memory::Memory(bytes(new_capacity)));

    device_memory::copy_device_to_device(
      new_smart_ptr.get(), 
      get(), 
      std::min(new_capacity, capacity));

    std::swap(capacity, new_capacity);
    std::swap(smart_ptr, new_smart_ptr);
  }

  /// Returns the number of elements
  size_t size() const {
    return capacity;
  }

  /// Returns the number of bytes needed to store the allocation
  size_t bytes() const {
    return bytes(capacity);
  }

  /// Returns a pointer to the managed allocation
  T* operator->() const { return smart_ptr.get()->ptr; }

  /// Copies a device-side memory allocation
  DeviceAllocation & operator=(DeviceAllocation const &p) {
    if (capacity != p.capacity) {
      reset(p.capacity);
    }
    device_memory::copy_device_to_device(get(), p.get(), capacity);
    return *this;
  }

  /// Move assignment
  DeviceAllocation & operator=(DeviceAllocation && p) {
    std::swap(capacity, p.capacity);
    std::swap(smart_ptr, p.smart_ptr);
    return *this;
  }

  /// Copies the entire allocation from another location in device memory.
  void copy_from_device(T const *ptr) const {
    copy_from_device(ptr, capacity);
  }

  /// Copies a given number of elements from device memory
  void copy_from_device(T const *ptr, size_t elements) const {
    device_memory::copy_device_to_device(get(), ptr, elements);
  }

  void copy_to_device(T *ptr) const {
    copy_to_device(ptr, capacity);
  }

  void copy_to_device(T *ptr, size_t elements) const {
    device_memory::copy_device_to_device(ptr, get(), elements);
  }

  void copy_from_host(T const *ptr) const {
    copy_from_host(ptr, capacity);
  }

  void copy_from_host(T const *ptr, size_t elements) const {
    device_memory::copy_to_device(get(), ptr, elements);
  }

  void copy_to_host(T *ptr) const {
    copy_to_host(ptr, capacity);
  }

  void copy_to_host(T *ptr, size_t elements) const {
    device_memory::copy_to_host(ptr, get(), elements); 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace device_memory {

/// Device allocation abstraction that tracks size and capacity
template <typename T>
using allocation = cutlass::DeviceAllocation<T>;

}  // namespace device_memory

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
