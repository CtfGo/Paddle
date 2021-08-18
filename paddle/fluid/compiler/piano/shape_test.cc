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

#include "paddle/fluid/compiler/piano/shape.h"
#include <utility>
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"

namespace paddle {
namespace piano {

class ShapeTest : public ::testing::Test {
 protected:
  const Shape scalar_ = Shape(note::F32, {});
  const Shape array_ = Shape(note::F32, {2, 3});
  const Shape array1_ = Shape(note::F32, {3, 6});

  const Shape result = Shape(note::F32, {2, 6});
  const Signature signature_ =
      Signature({array_, array1_}, {"arg1", "arg2"}, result);
};

TEST_F(ShapeTest, ShapeTransWithProto) {
  // Shape::ToProto on scalar
  auto&& scalar_proto = scalar_.ToProto();
  EXPECT_EQ(note::F32, scalar_proto.element_type());
  EXPECT_EQ(0, scalar_proto.dimensions_size());
  EXPECT_FALSE(scalar_proto.has_layout());
  EXPECT_FALSE(scalar_.IsTuple());
  EXPECT_TRUE(scalar_.IsArray());

  // construct Shape object from note::ShapeProto on scalar
  Shape scalar_from_proto(scalar_proto);
  EXPECT_EQ(scalar_from_proto.element_type(), scalar_.element_type());
  EXPECT_EQ(scalar_from_proto.dimensions().empty(),
            scalar_.dimensions().empty());
  EXPECT_EQ(scalar_from_proto.has_layout(), scalar_.has_layout());
  EXPECT_TRUE(scalar_from_proto.IsArray());

  // Shape::ToProto on array
  auto&& array_proto = array_.ToProto();
  EXPECT_EQ(note::F32, scalar_proto.element_type());
  EXPECT_EQ(2, array_proto.dimensions_size());
  EXPECT_EQ(2, array_proto.dimensions(0));
  EXPECT_EQ(3, array_proto.dimensions(1));
  EXPECT_FALSE(array_proto.has_layout());
  EXPECT_TRUE(array_.IsArray());

  // construct Shape object from note::ShapeProto on array
  Shape array_from_proto(array_proto);
  EXPECT_EQ(array_from_proto.element_type(), array_.element_type());
  EXPECT_EQ(array_from_proto.dimensions().size(), array_.dimensions().size());
  EXPECT_EQ(array_from_proto.has_layout(), array_.has_layout());
  EXPECT_TRUE(array_from_proto.IsArray());
}

TEST_F(ShapeTest, ShapeToString) {
  auto&& scalar_string = scalar_.ToString();
  ASSERT_EQ("f32[]{}", scalar_string);

  auto&& array_string = array_.ToString();
  ASSERT_EQ("f32[2, 3]{}", array_string);
}

TEST_F(ShapeTest, EqualToOther) {
  Shape s8_d23_no_layout(note::S32, {2, 3});
  Shape f32_d23_no_layout(note::F32, {2, 3});
  Shape f32_d45_no_layout(note::F32, {4, 5});
  Shape f32_d45_with_layout(note::F32, {4, 5});
  *f32_d45_with_layout.mutable_layout() = Layout({1, 0});
  Shape f32_d45_layout_reverse(note::F32, {4, 5});
  *f32_d45_layout_reverse.mutable_layout() = Layout({0, 1});

  Shape::CompareOption default_option;
  ASSERT_FALSE(s8_d23_no_layout.EqualTo(f32_d23_no_layout, {}));
  EXPECT_TRUE(s8_d23_no_layout.EqualTo(
      f32_d23_no_layout, Shape::CompareOption().IgnoreElementType()));
  ASSERT_FALSE(f32_d23_no_layout.EqualTo(f32_d45_no_layout, {}));
  EXPECT_TRUE(f32_d23_no_layout.EqualTo(
      f32_d45_no_layout, Shape::CompareOption().IgnoreDimensions()));
  ASSERT_FALSE(f32_d45_no_layout.EqualTo(f32_d45_with_layout, {}));
  EXPECT_TRUE(f32_d45_no_layout.EqualTo(f32_d45_with_layout,
                                        Shape::CompareOption().IgnoreLayout()));
  ASSERT_FALSE(f32_d45_with_layout.EqualTo(f32_d45_layout_reverse, {}));
}

TEST_F(ShapeTest, SignatureTransWithProto) {
  // Signature::ToProto
  auto&& signature_proto = signature_.ToProto();
  ASSERT_EQ(2, signature_proto.parameters_size());
  EXPECT_EQ(2, signature_proto.parameters(0).dimensions_size());
  EXPECT_EQ(2, signature_proto.parameters(1).dimensions_size());
  ASSERT_EQ(2, signature_proto.parameter_names_size());
  EXPECT_EQ("arg1", signature_proto.parameter_names(0));
  EXPECT_EQ("arg2", signature_proto.parameter_names(1));
  ASSERT_TRUE(signature_proto.has_result());
  EXPECT_EQ(2, signature_proto.result().dimensions_size());
  EXPECT_EQ(2, signature_proto.result().dimensions(0));
  EXPECT_EQ(6, signature_proto.result().dimensions(1));

  // construct Signature object from note::SignatureProto
  Signature signature_from_proto(signature_proto);
  ASSERT_EQ(2U, signature_from_proto.parameters().size());
  ASSERT_EQ(2U, signature_from_proto.parameter_names().size());
  ASSERT_EQ(2U, signature_from_proto.result().dimensions().size());
  EXPECT_EQ("f32[2, 6]{}", signature_from_proto.result().ToString());
}

TEST_F(ShapeTest, SignatureToString) {
  ASSERT_EQ("(arg1: f32[2, 3]{}, arg2: f32[3, 6]{}) -> f32[2, 6]{}",
            signature_.ToString());
}

}  // namespace piano
}  // namespace paddle
