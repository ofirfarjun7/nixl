/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * This file incorporates material from the DeepSeek project, licensed under the MIT License.
 * The modifications made by NVIDIA are licensed under the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: MIT AND Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "configs.cuh"
#include "exception.cuh"

#ifndef SETUP_LAUNCH_CONFIG
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[2]; \
    int __nixl_ep_device = 0; \
    cudaDeviceProp __nixl_ep_device_prop = {}; \
    CUDA_CHECK(cudaGetDevice(&__nixl_ep_device)); \
    CUDA_CHECK(cudaGetDeviceProperties(&__nixl_ep_device_prop, __nixl_ep_device)); \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 1; \
    if (__nixl_ep_device_prop.major >= 9) { \
        attr[1].id = cudaLaunchAttributeClusterDimension; \
        attr[1].val.clusterDim.x = ((num_sms) % 2 == 0 ? 2 : 1); \
        attr[1].val.clusterDim.y = 1; \
        attr[1].val.clusterDim.z = 1; \
        cfg.numAttrs = 2; \
    }
#endif

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#define SET_SHARED_MEMORY_FOR_TMA(kernel) \
EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
cfg.dynamicSmemBytes = smem_size;
#endif

#define SWITCH_NVL_RANKS(case_macro)                           \
    switch (num_nvl_ranks) {                                   \
        case 2:                                                \
            case_macro(2);                                     \
        case 4:                                                \
            case_macro(4);                                     \
        case 8:                                                \
            case_macro(8);                                     \
        default:                                               \
            EP_HOST_ASSERT(false and "Unsupported NVL ranks"); \
    }                                                          \
    while (false)

#define SWITCH_RDMA_RANKS(case_macro) \
    switch (num_ranks / NUM_MAX_NVL_PEERS) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        case 16: case_macro(16); \
        case 18: case_macro(18); \
        case 20: case_macro(20); \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } while (false)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2048: case_macro(2048); \
        case 2560: case_macro(2560); \
        case 3072: case_macro(3072); /*for gpt-oss*/ \
        case 4096: case_macro(4096); \
        case 5120: case_macro(5120); \
        case 6144: case_macro(6144); /* For qwen3 coder */ \
        case 7168: case_macro(7168); \
        case 8192: case_macro(8192); \
        default: EP_HOST_ASSERT(false and "Unsupported hidden"); \
    } while (false)
