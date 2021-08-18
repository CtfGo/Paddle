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
#include <algorithm>
#include <utility>
#include "paddle/fluid/compiler/piano/layout.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {

Shape::Shape(const note::ShapeProto& proto) {
  element_type_ = proto.element_type();
  dimensions_.reserve(proto.dimensions_size());
  dimensions_.insert(dimensions_.end(), proto.dimensions().begin(),
                     proto.dimensions().end());
  if (proto.has_layout()) {
    layout_ = Layout(proto.layout());
  }
}

note::ShapeProto Shape::ToProto() const {
  note::ShapeProto proto;
  proto.set_element_type(element_type());
  proto.mutable_dimensions()->Reserve(dimensions().size());
  for (const auto& dim : dimensions()) {
    proto.add_dimensions(dim);
  }
  if (has_layout()) {
    *proto.mutable_layout() = layout().ToProto();
  }

  return proto;
}

std::string Shape::ToString() const {
  auto dtype_name = note::ElementTypeProto_Name(element_type());
  std::transform(dtype_name.begin(), dtype_name.end(), dtype_name.begin(),
                 [](const auto& c) { return std::tolower(c); });

  std::vector<std::string> dim_names;
  std::transform(dimensions().begin(), dimensions().end(),
                 std::back_inserter(dim_names),
                 [](const auto& dim) { return std::to_string(dim); });

  return paddle::string::format_string(
      "%s[%s]%s", dtype_name.c_str(),
      paddle::string::join_strings(dim_names, ", ").c_str(),
      layout().ToString().c_str());
}

bool Shape::EqualTo(const Shape& other, const CompareOption& option) const {
  if (!option.ignore_element_type && element_type() != other.element_type()) {
    return false;
  }

  if (!option.ignore_dimensions && dimensions() != other.dimensions()) {
    return false;
  }

  if (!option.ignore_layout && layout() != other.layout()) {
    return false;
  }
  return true;
}

Signature::Signature(const note::SignatureProto& proto) {
  std::transform(proto.parameters().begin(), proto.parameters().end(),
                 std::back_inserter(parameters_),
                 [](const auto& shape_proto) { return Shape(shape_proto); });
  *mutable_result() = Shape(proto.result());
  parameter_names_.reserve(proto.parameter_names_size());
  parameter_names_.insert(parameter_names_.end(),
                          proto.parameter_names().begin(),
                          proto.parameter_names().end());
}

note::SignatureProto Signature::ToProto() const {
  note::SignatureProto proto;
  proto.mutable_parameters()->Reserve(parameters().size());
  for (const auto& shape : parameters()) {
    *proto.add_parameters() = shape.ToProto();
  }
  *proto.mutable_result() = result().ToProto();
  proto.mutable_parameter_names()->Reserve(parameter_names().size());
  for (const auto& name : parameter_names()) {
    proto.add_parameter_names(name);
  }
  return proto;
}

std::string Signature::ToString() const {
  std::vector<std::string> names;
  for (decltype(parameters().size()) i = 0; i < parameters().size(); ++i) {
    auto cur = paddle::string::format_string(
        "%s: %s", i < parameter_names().size() ? parameter_names().at(i).c_str()
                                               : "(unknown)",
        parameters().at(i).ToString().c_str());
    names.emplace_back(std::move(cur));
  }

  return paddle::string::format_string(
      "(%s) -> %s", paddle::string::join_strings(names, ", ").c_str(),
      result().ToString().c_str());
}

}  // namespace piano
}  // namespace paddle
