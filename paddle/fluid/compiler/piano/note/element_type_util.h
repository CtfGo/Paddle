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
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace piano {
namespace note {

// Map the given template parameter data type (eg, float)
// to the corresponding element proto type (eg, F32).
template <typename NativeT>
ElementTypeProto NativeToElementTypeProto() {
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map this native type to a proto type.");
  return note::INVALID_ELEMENT_TYPE;
}

// Declarations of specializations for each native type
// which correspond to a ElementTypeProto.
template <>
ElementTypeProto NativeToElementTypeProto<bool>();
template <>
ElementTypeProto NativeToElementTypeProto<int8_t>();
template <>
ElementTypeProto NativeToElementTypeProto<int16_t>();
template <>
ElementTypeProto NativeToElementTypeProto<int32_t>();
template <>
ElementTypeProto NativeToElementTypeProto<int64_t>();
template <>
ElementTypeProto NativeToElementTypeProto<uint8_t>();
template <>
ElementTypeProto NativeToElementTypeProto<uint16_t>();
template <>
ElementTypeProto NativeToElementTypeProto<uint32_t>();
template <>
ElementTypeProto NativeToElementTypeProto<uint64_t>();
template <>
ElementTypeProto NativeToElementTypeProto<platform::float16>();
template <>
ElementTypeProto NativeToElementTypeProto<float>();
template <>
ElementTypeProto NativeToElementTypeProto<double>();

// Map the given element proto type (eg, F32)
// to the corresponding native data type (eg, float).
template <ElementTypeProto>
struct ElementTypeProtoToNativeT;

// Declarations of specializations for each ElementTypeProto
// which correspond to a native type
template <>
struct ElementTypeProtoToNativeT<B1> {
  using type = bool;
};
template <>
struct ElementTypeProtoToNativeT<S8> {
  using type = int8_t;
};
template <>
struct ElementTypeProtoToNativeT<S16> {
  using type = int16_t;
};
template <>
struct ElementTypeProtoToNativeT<S32> {
  using type = int32_t;
};
template <>
struct ElementTypeProtoToNativeT<S64> {
  using type = int64_t;
};
template <>
struct ElementTypeProtoToNativeT<U8> {
  using type = uint8_t;
};
template <>
struct ElementTypeProtoToNativeT<U16> {
  using type = uint16_t;
};
template <>
struct ElementTypeProtoToNativeT<U32> {
  using type = uint32_t;
};
template <>
struct ElementTypeProtoToNativeT<U64> {
  using type = uint64_t;
};
template <>
struct ElementTypeProtoToNativeT<F16> {
  using type = platform::float16;
};
template <>
struct ElementTypeProtoToNativeT<F32> {
  using type = float;
};
template <>
struct ElementTypeProtoToNativeT<F64> {
  using type = double;
};

}  // namespace note
}  // namespace piano
}  // namespace paddle
