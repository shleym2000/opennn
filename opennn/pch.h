#pragma once

#define NDEBUG
#define EIGEN_NO_DEBUG

#define NUMERIC_LIMITS_MIN type(0.000001)

#define NOMINMAX
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <algorithm>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <codecvt>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <iterator>
#include <map>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <exception>
#include <memory>
#include <random>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <set>
#include <regex>
#include <sstream>
#include <omp.h>

#define EIGEN_USE_THREADS

#include "../eigen/Eigen/Core"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

//#define OPENNN_CUDA // Comment this line to disable cuda files

#ifdef OPENNN_CUDA

#include "../opennn/kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>
#include <cudnn.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call) do \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        string error_msg = std::string("CUDA Error: ") + cudaGetErrorString(err) + \
                                " in " + __FILE__ + ":" + std::to_string(__LINE__); \
        fprintf(stderr, "%s\n", error_msg.c_str()); \
        throw runtime_error(error_msg); \
    } \
} while(0)

#define CUDA_MALLOC_AND_REPORT(ptr, size)                                         \
    do {                                                                          \
        size_t free_before, free_after, total;                                    \
        CHECK_CUDA(cudaMemGetInfo(&free_before, &total));                         \
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (size)));         \
        CHECK_CUDA(cudaMemGetInfo(&free_after,  &total));                         \
                                                                                  \
        size_t bytes = free_before - free_after;                                  \
        if (bytes == 0) {                                                         \
            printf("cudaMalloc (%s):   reutilizado (%zu bytes solicitados)\n",    \
                   #ptr, (size_t)(size));                                         \
        } else {                                                                  \
            printf("cudaMalloc (%s):   %.6f MB  (%zu bytes)\n", #ptr,             \
                   bytes / (1024.0 * 1024.0), bytes);                             \
        }                                                                         \
    } while (0)


#define CHECK_CUBLAS(call) do \
{ \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) \
    { \
            string error_msg = std::string("CuBLAS Error code: ") + std::to_string(status) + \
              " in " + __FILE__ + ":" + std::to_string(__LINE__); \
            fprintf(stderr, "%s\n", error_msg.c_str()); \
            throw runtime_error(error_msg); \
    } \
} while(0)


#define CHECK_CUDNN(call) do \
{ \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) \
    { \
            string error_msg = std::string("cuDNN Error: ") + cudnnGetErrorString(status) + \
              " in " + __FILE__ + ":" + std::to_string(__LINE__); \
            fprintf(stderr, "%s\n", error_msg.c_str()); \
            throw runtime_error(error_msg); \
    } \
} while(0)
#endif

using namespace std;
using namespace Eigen;

using type = float;
using shape = vector<Index>;

#include "tinyxml2.h"

using namespace tinyxml2;

using Tensor1 = Tensor<type, 1>;
using Tensor2 = Tensor<type, 2>;
using Tensor3 = Tensor<type, 3>;
using Tensor4 = Tensor<type, 4>;
using Tensor5 = Tensor<type, 5>;

using TensorMap1 = TensorMap<Tensor<type, 1>, Aligned64>;
using TensorMap2 = TensorMap<Tensor<type, 2>, Aligned64>;
using TensorMap3 = TensorMap<Tensor<type, 3>, Aligned64>;
using TensorMap4 = TensorMap<Tensor<type, 4>, Aligned64>;

using ConstTensorMap1 = TensorMap<const Tensor<type, 1>, Aligned64>;
using ConstTensorMap2 = TensorMap<const Tensor<type, 2>, Aligned64>;
using ConstTensorMap3 = TensorMap<const Tensor<type, 3>, Aligned64>;
using ConstTensorMap4 = TensorMap<const Tensor<type, 4>, Aligned64>;

template<typename Base, typename T>
inline bool is_instance_of(const T* ptr)
{
    return dynamic_cast<const Base*>(ptr);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
