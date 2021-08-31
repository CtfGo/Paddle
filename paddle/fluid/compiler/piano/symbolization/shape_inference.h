/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/shape.h"

namespace paddle {
namespace piano {
namespace symbolization {
// Following functions are the resulting shape inference of meta operation.
// That is, for a given operation and input shapes, these functions
// infers what the resulting shape is for the operation,
// so that users can build computation via the meta_op API
// without specifying the result type or dimension

// inference for unary operation
Shape InferUnaryOpShape(note::OpCode opcode, const Shape& shape);

// inference for binary operation
Shape InferBinaryOpShape(note::OpCode opcode, const Shape& lhs,
                         const Shape& rhs);

// inference for broadcast operation
Shape InferBroadcastShape(const Shape& input_shape,
                          const std::vector<int64_t>& out_dimensions,
                          const std::vector<int64_t>& dimensions_alignment);

}  // namespace symbolization
}  // namespace piano
}  // namespace paddle
