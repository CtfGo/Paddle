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

// #include "paddle/fluid/compiler/piano/shape.h"
// #include "paddle/fluid/compiler/piano/shape_inference.h"
#include "paddle/fluid/compiler/piano/meta_op.h"
#include "paddle/fluid/compiler/piano/meta_op_util.h"
#include "paddle/fluid/compiler/piano/note_builder.h"

namespace paddle {
namespace piano {

Operand operator-(Operand x) { return Neg(x); }
Operand operator+(Operand x, Operand y) { return Add(x, y); }
Operand operator-(Operand x, Operand y) { return Sub(x, y); }
Operand operator*(Operand x, Operand y) { return Mul(x, y); }
Operand operator/(Operand x, Operand y) { return Div(x, y); }
Operand operator%(Operand x, Operand y) { return Rem(x, y); }

Operand operator~(Operand x) { return Not(x); }
Operand operator&(Operand x, Operand y) { return And(x, y); }
Operand operator|(Operand x, Operand y) { return Or(x, y); }
Operand operator^(Operand x, Operand y) { return Xor(x, y); }
Operand operator<<(Operand x, Operand y) { return ShiftLeft(x, y); }

Operand Not(Operand x) { return UnaryOp(note::Opcode::kNot, x); }
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

// Operand Rem(Operand x, Operand y, DimensionArray broadcast_dimensions) {
//   return BinaryOp(note::OpCode::kRem, x, y, broadcast_dimensions);
// }
//
// Operand ShiftLeft(Operand x, Operand y, DimensionArray broadcast_dimensions)
// {
//   return BinaryOp(note::OpCode::kShiftLeft, x, y, broadcast_dimensions);
// }

}  // namespace piano
}  // namespace paddle
