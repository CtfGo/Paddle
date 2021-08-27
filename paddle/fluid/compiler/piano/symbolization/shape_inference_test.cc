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

#include "paddle/fluid/compiler/piano/symbolization/shape_inference.h"
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"

namespace paddle {
namespace piano {
namespace symbolization {

TEST(ShapeInferenceTest, TestInferUnaryOpShape) {
  auto shape = Shape(note::F32, {2, 3});
  ASSERT_EQ(shape, InferUnaryOpShape(note::OpCode::kNegative, shape));
}

TEST(ShapeInferenceTest, TestInferBinaryOpShape) {
  auto lhs = Shape(note::F32, {2, 3});
  auto rhs = Shape(note::F32, {1, 3});
  ASSERT_EQ(lhs, InferBinaryOpShape(note::OpCode::kAdd, lhs, rhs));
}

TEST(ShapeInferenceTest, TestInferBroadcastShape) {
  // 1. check exceptional situation
  // 1.1 shape of operand should be a array
  auto subshape1 = Shape(note::F32, {1, 3});
  auto subshape2 = Shape(note::U64, {3, 6});
  ASSERT_THROW(InferBroadcastShape(
                   Shape(note::ELEMENT_TYPE_TUPLE, {2}, {subshape1, subshape2}),
                   {2, 3}, {0, 1}),
               paddle::platform::EnforceNotMet);
  // 1.2 out_dimensions equal to rank of operand
  ASSERT_THROW(InferBroadcastShape(subshape1, {2}, {0, 1}),
               paddle::platform::EnforceNotMet);
  // 1.3 dimensions_alignment equal to rank of operand
  ASSERT_THROW(InferBroadcastShape(subshape1, {5, 3}, {0}),
               paddle::platform::EnforceNotMet);
  // 1.4 the length of each out dimension are positive
  ASSERT_THROW(InferBroadcastShape(subshape1, {0, 3}, {0, 1}),
               paddle::platform::EnforceNotMet);
  // 1.5 dimensions_alignment valid
  ASSERT_THROW(InferBroadcastShape(subshape1, {5, 3}, {3, 4}),
               paddle::platform::EnforceNotMet);
  ASSERT_THROW(InferBroadcastShape(subshape1, {5, 2}, {0, 1}),
               paddle::platform::EnforceNotMet);
  ASSERT_THROW(InferBroadcastShape(subshape1, {5, 3}, {1, 0}),
               paddle::platform::EnforceNotMet);
  // 2. check return correct result
  auto res1 = InferBroadcastShape(subshape1, {5, 3}, {0, 1});
  ASSERT_EQ(Shape(note::F32, {5, 3}), res1);
  auto res2 = InferBroadcastShape(subshape2, {2, 3, 6}, {1, 2});
  ASSERT_EQ(Shape(note::U64, {2, 3, 6}), res2);
}

}  // namespace symbolization
}  // namespace piano
}  // namespace paddle
