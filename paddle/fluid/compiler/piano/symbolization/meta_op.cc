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

#include "paddle/fluid/compiler/piano/symbolization/meta_op.h"
#include <algorithm>
#include <numeric>
#include <utility>
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/symbolization/shape_inference.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace symbolization {

Operand Parameter(NoteBuilder* builder, int64_t parameter_index,
                  const Shape& shape, const std::string& name) {
  PADDLE_ENFORCE_GE(parameter_index, 0, platform::errors::InvalidArgument(
                                            "Parameter_index should be >= 0"));
  note::InstructionProto instr;
  instr.set_parameter_index(parameter_index);
  instr.set_name(name);
  *instr.mutable_shape() = shape.ToProto();
  return builder->AppendInstruction(std::move(instr), note::OpCode::kParameter,
                                    {});
}

Operand Broadcast(Operand x, const std::vector<int64_t>& out_dimensions,
                  const std::vector<int64_t>& dimensions_alignment) {
  // generate a default alignment for numpy's like broadcast operation
  std::vector<int64_t> to_right_alignment;
  if (dimensions_alignment.empty()) {
    PADDLE_ENFORCE_LE(x.Shape().Rank(), out_dimensions.size(),
                      platform::errors::InvalidArgument(
                          "Rank of operand should be less than output"));
    to_right_alignment.resize(x.Shape().Rank());
    std::iota(to_right_alignment.begin(), to_right_alignment.end(), 0);
    auto gap_len = out_dimensions.size() - x.Shape().Rank();
    // original operand is aligned to the rightmost of out_dimensions
    std::transform(to_right_alignment.begin(), to_right_alignment.end(),
                   to_right_alignment.begin(),
                   [gap_len](const auto& x) { return x + gap_len; });
  }

  const auto& alignment_array =
      dimensions_alignment.empty() ? to_right_alignment : dimensions_alignment;
  auto&& result_shape =
      InferBroadcastShape(x.Shape(), out_dimensions, alignment_array);

  note::InstructionProto instr;
  *instr.mutable_shape() = result_shape.ToProto();
  // fill the alignment array to kBroadcast attribute
  auto* attrs_map = instr.mutable_attrs();
  note::AttrValueProto attr_value;
  note::PopulateAttrValueProto(alignment_array, &attr_value);
  (*attrs_map)[note::kBroadcastAlignment] = attr_value;
  return x.Builder()->AppendInstruction(std::move(instr),
                                        note::OpCode::kBroadcast, {x});
}

Operand UnaryOp(note::OpCode unop, Operand x) {
  note::InstructionProto instr;
  auto&& shape = InferUnaryOpShape(unop, x.Shape());
  *instr.mutable_shape() = shape.ToProto();
  return x.Builder()->AppendInstruction(std::move(instr), unop, {x});
}

Operand operator-(Operand x) { return Neg(x); }
Operand operator~(Operand x) { return Not(x); }
Operand Neg(Operand x) { return UnaryOp(note::OpCode::kNegative, x); }
Operand Not(Operand x) { return UnaryOp(note::OpCode::kNot, x); }

Operand BinaryOp(note::OpCode binop, Operand x, Operand y) {
  // add broadcast if shape of the operands are not same
  x = x.Shape().Rank() < y.Shape().Rank() ? Broadcast(x, y.Shape().dimensions())
                                          : x;
  y = y.Shape().Rank() < x.Shape().Rank() ? Broadcast(y, x.Shape().dimensions())
                                          : y;
  // ensure shape are equal
  PADDLE_ENFORCE_EQ(x.Shape(), y.Shape(),
                    platform::errors::InvalidArgument(
                        "Shape of operands should be euqal on Binary Op"));

  note::InstructionProto instr;
  auto&& shape = InferBinaryOpShape(binop, x.Shape(), y.Shape());
  *instr.mutable_shape() = shape.ToProto();
  return x.Builder()->AppendInstruction(std::move(instr), binop, {x, y});
}

Operand operator+(Operand x, Operand y) { return Add(x, y); }
Operand operator-(Operand x, Operand y) { return Sub(x, y); }
Operand operator*(Operand x, Operand y) { return Mul(x, y); }
Operand operator/(Operand x, Operand y) { return Div(x, y); }
Operand operator&(Operand x, Operand y) { return And(x, y); }
Operand operator|(Operand x, Operand y) { return Or(x, y); }
Operand operator^(Operand x, Operand y) { return Xor(x, y); }

Operand Add(Operand x, Operand y) { return BinaryOp(note::OpCode::kAdd, x, y); }

Operand Sub(Operand x, Operand y) {
  return BinaryOp(note::OpCode::kSubtract, x, y);
}

Operand Mul(Operand x, Operand y) {
  return BinaryOp(note::OpCode::kMultiply, x, y);
}

Operand Div(Operand x, Operand y) {
  return BinaryOp(note::OpCode::kDivide, x, y);
}

Operand Max(Operand x, Operand y) {
  return BinaryOp(note::OpCode::kMaximum, x, y);
}

Operand Min(Operand x, Operand y) {
  return BinaryOp(note::OpCode::kMinimum, x, y);
}

Operand And(Operand x, Operand y) { return BinaryOp(note::OpCode::kAnd, x, y); }

Operand Or(Operand x, Operand y) { return BinaryOp(note::OpCode::kOr, x, y); }

Operand Xor(Operand x, Operand y) { return BinaryOp(note::OpCode::kXor, x, y); }

}  // namespace symbolization
}  // namespace piano
}  // namespace paddle
