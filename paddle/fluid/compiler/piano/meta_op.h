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
#include <vector>
#include "paddle/fluid/compiler/piano/note_builder.h"

namespace paddle {
namespace piano {

class Operand;
using DimensionArray = std::vector<int64_t>;

// initial instructions to retrieve data passed to the function
Operand Parameter(NoteBuilder* builder, int64_t parameter_number,
                  const Shape& shape, const string& name);
// Operand Constant(NoteBuilder* builder, const LiteralSlice& literal);

// unary op
Operand operator-(Operand x);
Operand operator~(Operand x);

// The broadcast semantic including two kinds of expanding operation on an
// array:
// 1. Adds dimensions to the array on the left, similarly to Numpy's rules,
//    here the "dimensions_alignment" can be empty.
// 2. Adds dimensions to the array among current dimensions,
//    using the "dimensions_alignment" parameter denotes that
//    which dimensions of the output array are aligned with the opeand
//    dimensions.
//
// `dimensions_alignment` are the dimensions to be broadcasting into, i.e., the
// i'th dimension of the operand is mapped to the dimensions_alignment[i]'th
// dimension of the output. This also requires that the i'th input dimension is
// either 1 or is the same as the output dimension it's broadcasting into.
//
// For example, say operand = {1, 2}, i.e., a 1D tensor in shape s32[2]; the
// output shape is s32[2,2]:
// - Specifying {1} as broadcast_dimension will generate output
//   {{1, 2},
//    {1, 2}}
// - On the other hand, specifying {0} as broadcast_dimension
//   will generate output
//   {{1 , 1},
//    {2 , 2}}
Operand Broadcast(Operand x, const DimensionArray& out_dimensions_size,
                  const DimensionArray& dimensions_alignment = {});

// binary op
Operand operator+(Operand x, Operand y);
Operand operator-(Operand x, Operand y);
Operand operator*(Operand x, Operand y);
Operand operator/(Operand x, Operand y);
Operand operator%(Operand x, Operand y);
Operand operator&(Operand x, Operand y);
Operand operator|(Operand x, Operand y);
Operand operator^(Operand x, Operand y);
Operand Add(Operand x, Operand y);
Operand Sub(Operand x, Operand y);
Operand Mul(Operand x, Operand y);
Operand Div(Operand x, Operand y);
Operand Max(Operand x, Operand y);
Operand Min(Operand x, Operand y);
Operand And(Operand x, Operand y);
Operand Or(Operand x, Operand y);
Operand Xor(Operand x, Operand y);

}  // namespace piano
}  // namespace paddle
