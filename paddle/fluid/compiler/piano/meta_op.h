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

namespace paddle {
namespace piano {

class Operand;

// unary op
Operand operator-(Operand x);
Operand operator~(Operand x);
// Operand Not(Operand x);

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
// Operand Rem(Operand x, Operand y, DimensionArray broadcast_dimensions = {});
// Operand ShiftLeft(Operand x, Operand y, DimensionArray broadcast_dimensions =
// {});

}  // namespace piano
}  // namespace paddle
