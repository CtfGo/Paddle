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
#include <vector>
#include "paddle/fluid/compiler/piano/note/attribute_key_defs.h"
#include "paddle/fluid/compiler/piano/note/element_type_util.h"
#include "paddle/fluid/compiler/piano/note/populate_attribute_value.h"
#include "paddle/fluid/compiler/piano/shape.h"
#include "paddle/fluid/compiler/piano/symbolization/note_builder.h"

namespace paddle {
namespace piano {

class Operand;
using DimensionArray = std::vector<int64_t>;

// initial instructions to retrieve data passed to the function
Operand Parameter(NoteBuilder* builder, int64_t parameter_index,
                  const Shape& shape, const std::string& name);

// define a constant containing 'value' with dimension 0 (scalar)
template <typename NativeT>
Operand ConstantD0(NoteBuilder* builder, NativeT value);

// unary op
Operand operator-(Operand x);
Operand operator~(Operand x);
Operand Neg(Operand x);
Operand Not(Operand x);

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
// For example, say operand = {1, 2}, i.e., a 1D tensor in shape s32[2] and
// expect the output shape is s32[2,2]:
// - Specifying {1} as dimensions_alignment will generate output
//   {{1, 2},
//    {1, 2}}
// - On the other hand, specifying {0} as dimensions_alignment
//   will generate output
//   {{1 , 1},
//    {2 , 2}}
Operand Broadcast(Operand x, const std::vector<int64_t>& out_dimensions,
                  const std::vector<int64_t>& dimensions_alignment = {});

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
Operand Rem(Operand x, Operand y);
Operand Or(Operand x, Operand y);
Operand Xor(Operand x, Operand y);

template <typename NativeT>
Operand ConstantD0(NoteBuilder* builder, NativeT value) {
  // construct shape
  Shape result_shape(note::NativeToElementTypeProto<NativeT>(), {});
  note::InstructionProto instr;
  *instr.mutable_shape() = result_shape.ToProto();
  // fill constant attribute
  auto* attrs_map = instr.mutable_attrs();
  note::AttrValueProto attr_value;
  note::PopulateAttrValueProtoD0(value, &attr_value);
  attrs_map->at(note::kConstantValue) = attr_value;
  return builder->AppendInstruction(std::move(instr), note::OpCode::kConstant,
                                    {});
}

}  // namespace piano
}  // namespace paddle
