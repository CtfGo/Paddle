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

#include "paddle/fluid/compiler/piano/note/element_type_util.h"
#include <type_traits>
#include "gtest/gtest.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"

namespace paddle {
namespace piano {
namespace note {

TEST(NativeToElementTypeProtoTest, Basic) {
  ASSERT_EQ(B1, NativeToElementTypeProto<bool>());
  ASSERT_EQ(S8, NativeToElementTypeProto<int8_t>());
  ASSERT_EQ(S16, NativeToElementTypeProto<int16_t>());
  ASSERT_EQ(S32, NativeToElementTypeProto<int32_t>());
  ASSERT_EQ(S64, NativeToElementTypeProto<int64_t>());
  ASSERT_EQ(U8, NativeToElementTypeProto<uint8_t>());
  ASSERT_EQ(U16, NativeToElementTypeProto<uint16_t>());
  ASSERT_EQ(U32, NativeToElementTypeProto<uint32_t>());
  ASSERT_EQ(U64, NativeToElementTypeProto<uint64_t>());
  ASSERT_EQ(F16, NativeToElementTypeProto<platform::float16>());
  ASSERT_EQ(F32, NativeToElementTypeProto<float>());
  ASSERT_EQ(F64, NativeToElementTypeProto<double>());
}

TEST(ElementTypeProtoToNativeTTest, Basic) {
  ASSERT_TRUE((std::is_same<bool, ElementTypeProtoToNativeT<B1>::type>::value));
  ASSERT_TRUE(
      (std::is_same<int8_t, ElementTypeProtoToNativeT<S8>::type>::value));
  ASSERT_TRUE(
      (std::is_same<int16_t, ElementTypeProtoToNativeT<S16>::type>::value));
  ASSERT_TRUE(
      (std::is_same<int32_t, ElementTypeProtoToNativeT<S32>::type>::value));
  ASSERT_TRUE(
      (std::is_same<int64_t, ElementTypeProtoToNativeT<S64>::type>::value));
  ASSERT_TRUE(
      (std::is_same<uint8_t, ElementTypeProtoToNativeT<U8>::type>::value));
  ASSERT_TRUE(
      (std::is_same<uint16_t, ElementTypeProtoToNativeT<U16>::type>::value));
  ASSERT_TRUE(
      (std::is_same<uint32_t, ElementTypeProtoToNativeT<U32>::type>::value));
  ASSERT_TRUE(
      (std::is_same<uint64_t, ElementTypeProtoToNativeT<U64>::type>::value));
  ASSERT_TRUE((std::is_same<platform::float16,
                            ElementTypeProtoToNativeT<F16>::type>::value));
  ASSERT_TRUE(
      (std::is_same<float, ElementTypeProtoToNativeT<F32>::type>::value));
  ASSERT_TRUE(
      (std::is_same<double, ElementTypeProtoToNativeT<F64>::type>::value));
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
