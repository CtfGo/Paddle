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

#include "paddle/fluid/compiler/piano/note/populate_attribute_value.h"
#include <type_traits>
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"

namespace paddle {
namespace piano {
namespace note {

TEST(PopulateAttrValueProtoTest, Basic) {
  AttrValueProto attr_value;
  // field 's'
  PopulateAttrValueProto<std::string>("yes", &attr_value);
  ASSERT_TRUE(attr_value.has_s());
  ASSERT_EQ("yes", attr_value.s());

  // field 'b'
  PopulateAttrValueProto(true, &attr_value);
  ASSERT_TRUE(attr_value.has_b());
  ASSERT_EQ(true, attr_value.b());

  // field 'i'
  PopulateAttrValueProto<int32_t>(110, &attr_value);
  ASSERT_TRUE(attr_value.has_i());
  ASSERT_EQ(110, attr_value.i());

  // field 'l'
  PopulateAttrValueProto<int64_t>(110, &attr_value);
  ASSERT_TRUE(attr_value.has_l());
  ASSERT_EQ(110, attr_value.l());

  // field 'f'
  PopulateAttrValueProto<float>(6.6, &attr_value);
  ASSERT_TRUE(attr_value.has_f());
  ASSERT_FLOAT_EQ(6.6, attr_value.f());

  // field 'd'
  PopulateAttrValueProto<double>(8.8, &attr_value);
  ASSERT_TRUE(attr_value.has_d());
  ASSERT_DOUBLE_EQ(8.8, attr_value.d());

  // field 'strings'
  PopulateAttrValueProto(std::vector<std::string>({"yes", "no"}), &attr_value);
  ASSERT_TRUE(attr_value.has_strings());
  ASSERT_EQ(2, attr_value.strings().value_size());
  ASSERT_EQ("yes", attr_value.strings().value(0));

  // field 'bools'
  PopulateAttrValueProto(std::vector<bool>({true, false, true}), &attr_value);
  ASSERT_TRUE(attr_value.has_bools());
  ASSERT_EQ(3, attr_value.bools().value_size());
  ASSERT_FALSE(attr_value.bools().value(1));

  // field 'ints'
  PopulateAttrValueProto(std::vector<int32_t>({110, 111, 666}), &attr_value);
  ASSERT_TRUE(attr_value.has_ints());
  ASSERT_EQ(3, attr_value.ints().value_size());
  ASSERT_EQ(666, attr_value.ints().value(2));

  // field 'longs'
  PopulateAttrValueProto(std::vector<int64_t>({666}), &attr_value);
  ASSERT_TRUE(attr_value.has_longs());
  ASSERT_EQ(1, attr_value.longs().value_size());
  ASSERT_EQ(666, attr_value.longs().value(0));

  // field 'floats'
  PopulateAttrValueProto(std::vector<float>({6.66, 8.88, 10.1}), &attr_value);
  ASSERT_TRUE(attr_value.has_floats());
  ASSERT_EQ(3, attr_value.floats().value_size());
  ASSERT_FLOAT_EQ(8.88, attr_value.floats().value(1));

  // field 'longs'
  PopulateAttrValueProto(std::vector<double>({8.88, 9.99}), &attr_value);
  ASSERT_TRUE(attr_value.has_doubles());
  ASSERT_EQ(2, attr_value.doubles().value_size());
  ASSERT_DOUBLE_EQ(9.99, attr_value.doubles().value(1));
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
