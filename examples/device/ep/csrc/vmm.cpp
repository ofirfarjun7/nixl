/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <stdexcept>

#include "config.hpp"
#include "vmm.hpp"

namespace {

constexpr const char *k_vmm_ctx = "vmm_region";

enum class cu_api_log_level { info, warning };

/** Log a non-fatal CUDA driver API failure (e.g. teardown). */
void
log_cu_api(CUresult status, cu_api_log_level level, const char *context,
           const char *message) noexcept {
    if (status != CUDA_SUCCESS) {
        const char *cuda_msg = nullptr;
        if (cuGetErrorString(status, &cuda_msg) != CUDA_SUCCESS || cuda_msg == nullptr) {
            cuda_msg = "unknown CUDA driver error";
        }
        const char *level_str =
            (level == cu_api_log_level::warning) ? "WARNING" : "INFO";
        std::cerr << level_str << ": " << context << " " << message << ": " << cuda_msg << '\n';
    }
}

} // namespace

namespace nixl_ep {

struct cuda_alloc_ctx {
    bool fabric_supported;
    CUmemAllocationProp prop;
    size_t granularity;
    CUdevice device;
    CUmemAccessDesc access_desc = {};

    cuda_alloc_ctx() : fabric_supported(false), prop({}), granularity(0) {
        int version;

        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
            throw std::runtime_error("CUDA device should be set before creating a vmm_region");
        }

        if (cuDriverGetVersion(&version) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA driver version");
        }

        if (version < 11000) {
            return; /* too old — fall back to cudaMalloc */
        }

        int fab = 0;
        if ((cuDeviceGetAttribute(&fab,
                                  CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                                  device) != CUDA_SUCCESS) ||
            (!fab)) {
            return; /* no fabric — fall back to cudaMalloc */
        }

        int rdma_vmm_supported = 0;
        if (cuDeviceGetAttribute(&rdma_vmm_supported,
                                 CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
                                 device) != CUDA_SUCCESS) {
            throw std::runtime_error(
                "Failed to query GPUDirect RDMA with VMM support attribute");
        }

        if (!rdma_vmm_supported) {
            std::cerr << "DIAG: " << k_vmm_ctx
                      << " - GPUDirect RDMA with CUDA VMM not supported; falling back to "
                         "cuMemAlloc\n";
            return;
        }

        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        prop.allocFlags.gpuDirectRDMACapable = 1;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        if (cuMemGetAllocationGranularity(
                &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS) {
            throw std::runtime_error("Failed to get CUDA allocation granularity");
        }

        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        fabric_supported = true;
    }
};

bool
vmm_region::vmmFabricAllocation(cuda_alloc_ctx &ctx, size_t size) {

    if (!ctx.fabric_supported) {
        return false;
    }

    size_ = nixl_ep::align_up<size_t>(size, ctx.granularity);

    CUresult st = cuMemCreate(&handle_, size_, &ctx.prop, 0);
    if (st != CUDA_SUCCESS) {
        log_cu_api(st, cu_api_log_level::warning, k_vmm_ctx, "failed to create CUDA VMM allocation");
        return false;
    }

    st = cuMemAddressReserve(&ptr_, size_, 0, 0, 0);
    if (st != CUDA_SUCCESS) {
        log_cu_api(st, cu_api_log_level::warning, k_vmm_ctx, "failed to reserve CUDA virtual address");
        release();
        return false;
    }

    st = cuMemMap(ptr_, size_, 0, handle_, 0);
    if (st != CUDA_SUCCESS) {
        log_cu_api(st, cu_api_log_level::warning, k_vmm_ctx, "failed to map CUDA VMM memory");
        release();
        return false;
    }
    vmm_mapped_ = true;

    st = cuMemSetAccess(ptr_, size_, &ctx.access_desc, 1);
    if (st != CUDA_SUCCESS) {
        log_cu_api(st, cu_api_log_level::warning, k_vmm_ctx, "failed to set CUDA memory access");
        release();
        return false;
    }

    return true;
}

void
vmm_region::release() noexcept {
    if (is_cuda_malloc_) {
        if (ptr_) {
            warn_cu_api(cuMemFree(ptr_), k_vmm_ctx, "cuMemFree");
        }
        ptr_ = 0;
        return;
    }

    if (vmm_mapped_) {
        log_cu_api(cuMemUnmap(ptr_, size_), cu_api_log_level::warning, k_vmm_ctx, "failed to cuMemUnmap");
        vmm_mapped_ = false;
    }
    if (ptr_) {
        log_cu_api(cuMemAddressFree(ptr_, size_), cu_api_log_level::warning, k_vmm_ctx,
                   "cuMemAddressFree");
        ptr_ = 0;
    }
    if (handle_) {
        log_cu_api(cuMemRelease(handle_), cu_api_log_level::warning, k_vmm_ctx, "failed to cuMemRelease");
        handle_ = 0;
    }
}

vmm_region::~vmm_region() {
    release();
}

vmm_region::vmm_region(size_t size) {
    if (size == 0) {
        throw std::invalid_argument("vmm_region: size must be non-zero");
    }

    static cuda_alloc_ctx ctx;
    if (vmmFabricAllocation(ctx, size)) {
        return;
    } else if (ctx.fabric_supported) {
        std::cerr << "INFO: " << k_vmm_ctx
                      << " - CUDA VMM fabric allocation failed; falling back to "
                         "cuMemAlloc\n";
    }

    size_ = size;
    is_cuda_malloc_ = true;
    if (cuMemAlloc(&ptr_, size) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc fallback failed");
    }
}

} // namespace nixl_ep
