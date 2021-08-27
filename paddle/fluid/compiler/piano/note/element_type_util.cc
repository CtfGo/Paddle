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

namespace paddle {
namespace piano {
namespace note {

template <>
ElementTypeProto NativeToElementTypeProto<bool>() {
  return B1;
}

template <>
ElementTypeProto NativeToElementTypeProto<int8_t>() {
  return S8;
}

template <>
ElementTypeProto NativeToElementTypeProto<int16_t>() {
  return S16;
}

template <>
ElementTypeProto NativeToElementTypeProto<int32_t>() {
  return S32;
}

template <>
ElementTypeProto NativeToElementTypeProto<int64_t>() {
  return S64;
}

template <>
ElementTypeProto NativeToElementTypeProto<uint8_t>() {
  return U8;
}

template <>
ElementTypeProto NativeToElementTypeProto<uint16_t>() {
  return U16;
}

template <>
ElementTypeProto NativeToElementTypeProto<uint32_t>() {
  return U32;
}

template <>
ElementTypeProto NativeToElementTypeProto<uint64_t>() {
  return U64;
}

template <>
ElementTypeProto NativeToElementTypeProto<platform::float16>() {
  return F16;
}

template <>
ElementTypeProto NativeToElementTypeProto<float>() {
  return F32;
}

template <>
ElementTypeProto NativeToElementTypeProto<double>() {
  return F64;
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
