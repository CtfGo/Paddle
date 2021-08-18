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

#include "paddle/fluid/compiler/piano/layout.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {

class LayoutTest : public ::testing::Test {
 protected:
  const Layout layout_ = Layout({3, 2, 1, 0});
};

TEST_F(LayoutTest, LayoutTransWithProto) {
  // Layout::ToProto and construct from a note::LayoutProto
  auto&& layout_proto = layout_.ToProto();
  ASSERT_EQ(4, layout_proto.minor_to_major_size());
  EXPECT_EQ(1, layout_proto.minor_to_major(2));

  Layout layout_from_proto(layout_proto);
  ASSERT_TRUE(layout_from_proto.Valid());
  ASSERT_EQ(4U, layout_from_proto.minor_to_major().size());
}

TEST_F(LayoutTest, LayoutToString) {
  ASSERT_EQ("{3, 2, 1, 0}", layout_.ToString());
}

}  // namespace piano
}  // namespace paddle
