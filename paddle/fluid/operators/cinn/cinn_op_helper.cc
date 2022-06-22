// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/cinn/cinn_op_helper.h"

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/device_context.h"

#ifdef PADDLE_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace paddle::operators::details {

#ifdef PADDLE_WITH_CUDA

void CUDART_CB ReleaseScope(void* data) {
  auto* temp_scope = static_cast<framework::Scope*>(data);
  delete temp_scope;
}

template <>
void ReleaseResource<platform::CUDADeviceContext>(
    const std::vector<void*>& resources, void* stream) {
  PADDLE_ENFORCE_GPU_SUCCESS(cudaLaunchHostFunc(
      static_cast<gpuStream_t>(stream), ReleaseScope, resources[0]));
}

template <>
void* GetStream<platform::CUDADeviceContext>(
    const framework::ExecutionContext& ctx) {
  const auto& dev_ctx =
      ctx.template device_context<platform::CUDADeviceContext>();
  return dev_ctx.stream();
}
#endif

}  // namespace paddle::operators::details
