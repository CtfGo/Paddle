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

#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {
namespace note {

// The order should be as same as note.proto
using AttrType =
    boost::variant<boost::blank, std::string, bool, std::int32_t, std::int64_t,
                   float, double, std::vector<std::string>, std::vector<bool>,
                   std::vector<std::int32_t>, std::vector<std::int64_t>,
                   std::vector<float>, std::vector<double>>;

using MapType = std::unordered_map<std::string, AttrType>;

using ProtoMapType = google::protobuf::Map<std::string, AttrValueProto>;

struct AttrToString : public boost::static_visitor<std::string> {
  std::string operator()(const boost::blank &) const { return "blank"; }
  std::string operator()(const std::string &val) const {
    return string::format_string("\"%s\"", val.c_str());
  }
  std::string operator()(const bool &val) const {
    return val ? "true" : "false";
  }
  std::string operator()(const std::int32_t &val) const {
    return string::format_string("%s_i", std::to_string(val).c_str());
  }
  std::string operator()(const std::int64_t &val) const {
    return string::format_string("%s_l", std::to_string(val).c_str());
  }
  std::string operator()(const float &val) const {
    return string::format_string("%s_f", std::to_string(val).c_str());
  }
  std::string operator()(const double &val) const {
    return string::format_string("%s_d", std::to_string(val).c_str());
  }
  std::string operator()(const std::vector<std::string> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const std::string &v) {
                     return string::format_string("\"%s\"", v.c_str());
                   });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }
  std::string operator()(const std::vector<bool> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const bool &v) { return v ? "true" : "false"; });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }
  std::string operator()(const std::vector<std::int32_t> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const std::int32_t &v) {
                     return string::format_string("%s_i",
                                                  std::to_string(v).c_str());
                   });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }
  std::string operator()(const std::vector<std::int64_t> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const std::int64_t &v) {
                     return string::format_string("%s_l",
                                                  std::to_string(v).c_str());
                   });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }
  std::string operator()(const std::vector<float> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const float &v) {
                     return string::format_string("%s_f",
                                                  std::to_string(v).c_str());
                   });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }
  std::string operator()(const std::vector<double> &val) const {
    std::vector<std::string> lexical;
    lexical.reserve(val.size());
    std::transform(val.cbegin(), val.cend(), std::back_inserter(lexical),
                   [](const double &v) {
                     return string::format_string("%s_d",
                                                  std::to_string(v).c_str());
                   });
    return string::format_string("{%s}",
                                 string::join_strings(lexical, ", ").c_str());
  }

  template <typename T>
  std::string operator()(const T &val) const {
    return "unknown";
  }
};

#define _ForEachAttrTypeHelper_(callback, cpp_type, proto_type) \
  callback(cpp_type,                                            \
           ::paddle::piano::note::AttrValueProto::ValueCase::proto_type)

#define _ForEachAttrType_(callback)                                      \
  _ForEachAttrTypeHelper_(callback, std::string, kS);                    \
  _ForEachAttrTypeHelper_(callback, bool, kB);                           \
  _ForEachAttrTypeHelper_(callback, std::int32_t, kI);                   \
  _ForEachAttrTypeHelper_(callback, std::int64_t, kL);                   \
  _ForEachAttrTypeHelper_(callback, float, kF);                          \
  _ForEachAttrTypeHelper_(callback, double, kD);                         \
  _ForEachAttrTypeHelper_(callback, std::vector<std::string>, kStrings); \
  _ForEachAttrTypeHelper_(callback, std::vector<bool>, kBools);          \
  _ForEachAttrTypeHelper_(callback, std::vector<std::int32_t>, kInts);   \
  _ForEachAttrTypeHelper_(callback, std::vector<std::int64_t>, kLongs);  \
  _ForEachAttrTypeHelper_(callback, std::vector<float>, kFloats);        \
  _ForEachAttrTypeHelper_(callback, std::vector<double>, kDoubles);

template <typename Visitor>
inline void VisitAttrType(::paddle::piano::note::AttrValueProto::ValueCase type,
                          Visitor visitor) {
#define VisitAttrTypeCallback(cpp_type, proto_type) \
  do {                                              \
    if (type == proto_type) {                       \
      visitor.template apply<cpp_type>();           \
      return;                                       \
    }                                               \
  } while (0)

  _ForEachAttrType_(VisitAttrTypeCallback);
#undef VisitAttrTypeCallback
  PADDLE_THROW(
      platform::errors::Unimplemented("Invalid attribute type %d.", type));
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
