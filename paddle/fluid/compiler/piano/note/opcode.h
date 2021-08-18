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

#include <cstdint>
#include <string>

namespace paddle {
namespace piano {
namespace note {

// HANDLE(ID, DESC, PARAM_NUM, ...)
#define OPCODE_HANDLER(HANDLE)                          \
  HANDLE(Constant, "constant", 0)                       \
  HANDLE(Parameter, "parameter", 0)                     \
  HANDLE(BatchNormGrad, "batch_norm_grad", 5)           \
  HANDLE(BatchNormInference, "batch_norm_inference", 5) \
  HANDLE(BatchNormTraining, "batch_norm_training", 3)   \
  HANDLE(Convolution, "convolution", 2)                 \
  HANDLE(Dot, "dot", 2)                                 \
  HANDLE(Broadcast, "broadcast", 1)                     \
  HANDLE(Cast, "cast", 1)                               \
  HANDLE(Copy, "copy", 1)                               \
  HANDLE(Exp, "exp", 1)                                 \
  HANDLE(Log, "log", 1)                                 \
  HANDLE(Negative, "negative", 1)                       \
  HANDLE(Not, "not", 1)                                 \
  HANDLE(Reshape, "reshape", 1)                         \
  HANDLE(Reverse, "reverse", 1)                         \
  HANDLE(Rsqrt, "rsqrt", 1)                             \
  HANDLE(Slice, "slice", 1)                             \
  HANDLE(Sqrt, "sqrt", 1)                               \
  HANDLE(Transpose, "transpose", 1)                     \
  HANDLE(Add, "add", 2)                                 \
  HANDLE(And, "and", 2)                                 \
  HANDLE(Compare, "compare", 2)                         \
  HANDLE(Divide, "divide", 2)                           \
  HANDLE(Maximum, "maximum", 2)                         \
  HANDLE(Minimum, "minimum", 2)                         \
  HANDLE(Multiply, "multiply", 2)                       \
  HANDLE(Or, "or", 2)                                   \
  HANDLE(Subtract, "subtract", 2)                       \
  HANDLE(Xor, "xor", 2)                                 \
  HANDLE(Select, "select", 3)                           \
  HANDLE(Concatenate, "concatenate", kVariadicParamNum) \
  HANDLE(Reduce, "reduce", kVariadicParamNum)           \
  HANDLE(Rng, "rng", kVariadicParamNum)                 \
  HANDLE(Sort, "sort", kVariadicParamNum)               \
  HANDLE(Tuple, "tuple", kVariadicParamNum)

constexpr std::int32_t kVariadicParamNum = -1;

enum class OpCode : std::uint32_t {
#define HANDLE_ENUM(enum_id, ...) k##enum_id,
  OPCODE_HANDLER(HANDLE_ENUM)
#undef HANDLE_ENUM
      kNumOps
};

OpCode GetOpCode(const std::string& op_name);

const std::string& GetOpName(OpCode code);

int GetOpParamNum(OpCode code);

}  // namespace note
}  // namespace piano
}  // namespace paddle
