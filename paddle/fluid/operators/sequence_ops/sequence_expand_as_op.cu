/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>

#include "paddle/fluid/operators/sequence_ops/sequence_expand_as_op.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
static __global__ void sequence_expand_as_kernel(const T *in_data,
                                                 const size_t *expand_offset,
                                                 const size_t src_hight,
                                                 const size_t src_widht,
                                                 T *out_data) {
  for (int h_id = blockIdx.x; h_id < src_hight; h_id += gridDim.x) {
    int span = expand_offset[h_id + 1] - expand_offset[h_id];
    if (span == 0) continue;
    const T *src = in_data + h_id * src_widht;
    for (int w_id = threadIdx.x; w_id < src_widht; w_id += blockDim.x) {
      T ele = src[w_id];
      int offset = expand_offset[h_id] * src_widht;
      for (int k = 0; k < span; ++k) {
        out_data[offset + k * src_widht + w_id] = ele;
      }
    }
  }
}

template <typename T>
static __global__ void sequence_expand_as_grad_kernel(
    const T *dout_data,
    const size_t *expand_offset,
    const size_t dst_hight,
    const size_t dst_width,
    T *dx_data) {
  for (int h_id = blockIdx.x; h_id < dst_hight; h_id += gridDim.x) {
    T *dst = dx_data + h_id * dst_width;
    int span = expand_offset[h_id + 1] - expand_offset[h_id];

    for (int w_id = threadIdx.x; w_id < dst_width; w_id += blockDim.x) {
      T result = 0;
      for (int k = 0; k < span; ++k) {
        int offset = (expand_offset[h_id] + k) * dst_width;
        const T *src = dout_data + offset;
        result += src[w_id];
      }
      dst[w_id] = result;
    }
  }
}

template <typename T>
struct SequenceExpandAsFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext &context,
                  const phi::DenseTensor &x,
                  const phi::Vector<size_t> &ref_lod, /*expand referenced lod*/
                  phi::DenseTensor *out) {
    int height = x.dims()[0];
    int width = phi::product(x.dims()) / height;

    const int kThreadsPerBlock = 1024;
    int thread_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock) {  // block_cols is aligned by 32.
      thread_x = ((width + 31) >> 5) << 5;
    }

    int max_threads = context.GetMaxPhysicalThreadCount();
    int block_x = std::max(max_threads / thread_x, 1);

    dim3 block_size(thread_x);
    dim3 grid_size(block_x);
    phi::MixVector<size_t> mixv_ref_lod(&ref_lod);
    sequence_expand_as_kernel<<<grid_size, block_size, 0, context.stream()>>>(
        x.data<T>(),
        mixv_ref_lod.CUDAData(context.GetPlace()),
        height,
        width,
        out->mutable_data<T>(context.GetPlace()));
  }
};

template <typename T>
struct SequenceExpandAsGradFunctor<phi::GPUContext, T> {
  void operator()(const phi::GPUContext &context,
                  const phi::DenseTensor &dout,
                  const phi::Vector<size_t> &ref_lod, /*expand based lod*/
                  phi::DenseTensor *dx) {
    int height = dx->dims()[0];
    int width = phi::product(dx->dims()) / height;

    const int kThreadsPerBlock = 1024;
    int thread_x = kThreadsPerBlock;
    if (width < kThreadsPerBlock) {  // block_cols is aligned by 32.
      thread_x = ((width + 31) >> 5) << 5;
    }

    int max_threads = context.GetMaxPhysicalThreadCount();
    int block_x = std::max(max_threads / thread_x, 1);

    dim3 block_size(thread_x);
    dim3 grid_size(block_x);
    phi::MixVector<size_t> mixv_ref_lod(&ref_lod);
    sequence_expand_as_grad_kernel<<<grid_size,
                                     block_size,
                                     0,
                                     context.stream()>>>(
        dout.data<T>(),
        mixv_ref_lod.CUDAData(context.GetPlace()),
        height,
        width,
        dx->mutable_data<T>(context.GetPlace()));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
PD_REGISTER_STRUCT_KERNEL(sequence_expand_as,
                          GPU,
                          ALL_LAYOUT,
                          ops::SequenceExpandAsKernel,
                          float,
                          double,
                          int,
                          int64_t) {}
PD_REGISTER_STRUCT_KERNEL(sequence_expand_as_grad,
                          GPU,
                          ALL_LAYOUT,
                          ops::SequenceExpandAsGradKernel,
                          float,
                          double,
                          int,
                          int64_t) {}
