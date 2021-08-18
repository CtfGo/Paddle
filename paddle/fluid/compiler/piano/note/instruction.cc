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

#include "paddle/fluid/compiler/piano/note/instruction.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "paddle/fluid/compiler/piano/note/function.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/note/type_traits.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace piano {
namespace note {

Instruction::Instruction(
    const InstructionProto& proto,
    const std::unordered_map<std::int64_t, Instruction*>& instr_index,
    const std::unordered_map<std::int64_t, Function*>& func_index)
    : name_(proto.name()), shape_(proto.shape()), global_id_(proto.id()) {
  opcode_ = GetOpCode(proto.opcode());
  // set operands
  operands_.resize(proto.operand_ids_size());
  std::transform(
      proto.operand_ids().begin(), proto.operand_ids().end(), operands_.begin(),
      [&instr_index](std::int64_t index) { return instr_index.at(index); });

  // set call functions
  call_functions_.resize(proto.call_function_ids_size());
  std::transform(
      proto.call_function_ids().begin(), proto.call_function_ids().end(),
      call_functions_.begin(),
      [&func_index](std::int64_t index) { return func_index.at(index); });

  // add control dependency
  for (auto id : proto.control_predecessor_ids()) {
    PADDLE_ENFORCE_EQ(instr_index.at(id)->parent(), parent(),
                      platform::errors::PreconditionNotMet(
                          "The instruction and its dependent instruction are "
                          "not in the same function."));
    auto& successors = instr_index.at(id)->ctrl_successors_;
    if (std::find(successors.begin(), successors.end(), this) ==
        successors.end()) {
      successors.push_back(this);
    }
    auto& predecessors = this->ctrl_predecessors_;
    if (std::find(predecessors.begin(), predecessors.end(),
                  instr_index.at(id)) == predecessors.end()) {
      predecessors.push_back(instr_index.at(id));
    }
  }

  // set parameter number
  if (proto.has_parameter_number()) {
    PADDLE_ENFORCE_EQ(proto.parameter_number(), operands_.size(),
                      platform::errors::PreconditionNotMet(
                          "The number of operands(%ld) is not equal to the "
                          "parameter_number(%zu) in proto.",
                          proto.parameter_number(), operands_.size()));
    parameter_number_ = proto.parameter_number();
  } else {
    parameter_number_ = static_cast<std::int64_t>(operands_.size());
  }

  // set attrs
  const auto& attrs_map = proto.attrs();
  for (auto it = attrs_map.begin(); it != attrs_map.end(); it++) {
    attrs_.emplace(it->first, GetAttrValue(it->second));
  }
}

InstructionProto Instruction::ToProto() const {
  InstructionProto proto;

  // serialize basic info
  proto.set_name(name_);
  proto.set_opcode(GetOpName(opcode_));
  proto.set_id(global_id_);
  proto.set_parameter_number(parameter_number_);

  // serialize shape info
  *proto.mutable_shape() = shape_.ToProto();

  // serialize operands info
  std::for_each(operands_.cbegin(), operands_.cend(),
                [&proto](const Instruction* instr) {
                  proto.add_operand_ids(instr->global_id());
                });

  // serialize control dependency info
  std::for_each(ctrl_predecessors_.cbegin(), ctrl_predecessors_.cend(),
                [&proto](const Instruction* instr) {
                  proto.add_control_predecessor_ids(instr->global_id());
                });

  // serialize call functions info
  std::for_each(call_functions_.cbegin(), call_functions_.cend(),
                [&proto](const Function* func) {
                  proto.add_call_function_ids(func->global_id());
                });

  // serialize attrs map
  auto* attrs_map = proto.mutable_attrs();
  for (auto it = attrs_.begin(); it != attrs_.end(); it++) {
    auto val_proto = GetAttrProto(it->first);
    attrs_map->insert(ProtoMapType::value_type(it->first, val_proto));
  }

  return proto;
}

std::string Instruction::ToString() const {
  // get string values(shape&name) of operands this instruction uses
  std::vector<std::string> operand_strs;
  operand_strs.reserve(operands_.size());
  std::transform(operands_.cbegin(), operands_.cend(),
                 std::back_inserter(operand_strs),
                 [](const Instruction* operand) {
                   return string::format_string(
                       "%s %s%s", operand->shape().ToString().c_str(), "%",
                       operand->name().c_str());
                 });
  // get function names this instruction directly calls
  std::string fns_str = "";
  if (call_functions_.size() != 0) {
    std::vector<std::string> used_fns;
    used_fns.reserve(call_functions_.size());
    std::transform(call_functions_.cbegin(), call_functions_.cend(),
                   std::back_inserter(used_fns),
                   [](const Function* fn) { return fn->name(); });
    fns_str = string::format_string(
        ", call_functions={%s}", string::join_strings(used_fns, ", ").c_str());
  }
  // get string values of attributes
  std::vector<std::string> attr_strs;
  AttrToString to_str;
  for (auto it = attrs_.cbegin(); it != attrs_.cend(); it++) {
    attr_strs.emplace_back(string::format_string(
        "%s=%s", it->first.c_str(),
        boost::apply_visitor(to_str, it->second).c_str()));
  }
  std::sort(attr_strs.begin(), attr_strs.end());

  return string::format_string(
      "%s%s = %s %s(%s)%s, %s", "%", name_.c_str(), shape_.ToString().c_str(),
      GetOpName(opcode_).c_str(),
      string::join_strings(operand_strs, ", ").c_str(), fns_str.c_str(),
      string::join_strings(attr_strs, ", ").c_str());
}

void Instruction::Accept(backends::NoteVisitorBase* visitor) {
  switch (opcode_) {
#define HANDLE_VISIT(enum_id, op_name, ...) \
  case OpCode::k##enum_id:                  \
    return visitor->Visit##enum_id(*this);
    OPCODE_HANDLER(HANDLE_VISIT)
#undef HANDLE_VISIT
    default:
      PADDLE_THROW(
          platform::errors::NotFound("Invalid operator code (opcode = %u) !",
                                     static_cast<std::uint32_t>(opcode_)));
  }
}

AttrValueProto Instruction::GetAttrProto(const std::string& attr_name) const {
  AttrValueProto val_proto;
  auto value_type =
      static_cast<AttrValueProto::ValueCase>(attrs_.at(attr_name).which());
  switch (value_type) {
    case AttrValueProto::ValueCase::kS: {
      val_proto.set_s(GetAttr<std::string>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kB: {
      val_proto.set_b(GetAttr<bool>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kI: {
      val_proto.set_i(GetAttr<std::int32_t>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kL: {
      val_proto.set_l(GetAttr<std::int64_t>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kF: {
      val_proto.set_f(GetAttr<float>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kD: {
      val_proto.set_d(GetAttr<double>(attr_name));
      break;
    }
    case AttrValueProto::ValueCase::kStrings: {
      auto val = GetAttr<std::vector<std::string>>(attr_name);
      auto* strings = val_proto.mutable_strings()->mutable_value();
      std::for_each(val.cbegin(), val.cend(),
                    [&strings](const std::string& s) { *strings->Add() = s; });
      break;
    }
    case AttrValueProto::ValueCase::kBools: {
      auto val = GetAttr<std::vector<bool>>(attr_name);
      auto* bools = val_proto.mutable_bools()->mutable_value();
      bools->Resize(val.size(), false);
      std::copy(val.cbegin(), val.cend(), bools->begin());
      break;
    }
    case AttrValueProto::ValueCase::kInts: {
      auto val = GetAttr<std::vector<std::int32_t>>(attr_name);
      auto* ints = val_proto.mutable_ints()->mutable_value();
      ints->Resize(val.size(), 0);
      std::copy(val.cbegin(), val.cend(), ints->begin());
      break;
    }
    case AttrValueProto::ValueCase::kLongs: {
      auto val = GetAttr<std::vector<std::int64_t>>(attr_name);
      auto* longs = val_proto.mutable_longs()->mutable_value();
      longs->Resize(val.size(), 0);
      std::copy(val.cbegin(), val.cend(), longs->begin());
      break;
    }
    case AttrValueProto::ValueCase::kFloats: {
      auto val = GetAttr<std::vector<float>>(attr_name);
      auto* floats = val_proto.mutable_floats()->mutable_value();
      floats->Resize(val.size(), 0);
      std::copy(val.cbegin(), val.cend(), floats->begin());
      break;
    }
    case AttrValueProto::ValueCase::kDoubles: {
      auto val = GetAttr<std::vector<double>>(attr_name);
      auto* doubles = val_proto.mutable_doubles()->mutable_value();
      doubles->Resize(val.size(), 0);
      std::copy(val.cbegin(), val.cend(), doubles->begin());
      break;
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Invalid attribute type %d.",
                                                 value_type));
  }

  return val_proto;
}

AttrType Instruction::GetAttrValue(const AttrValueProto& value_proto) const {
  switch (value_proto.value_case()) {
    case AttrValueProto::ValueCase::kS:
      return value_proto.s();
    case AttrValueProto::ValueCase::kB:
      return value_proto.b();
    case AttrValueProto::ValueCase::kI:
      return value_proto.i();
    case AttrValueProto::ValueCase::kL:
      return value_proto.l();
    case AttrValueProto::ValueCase::kF:
      return value_proto.f();
    case AttrValueProto::ValueCase::kD:
      return value_proto.d();
    case AttrValueProto::ValueCase::kStrings: {
      const auto& val = value_proto.strings().value();
      return std::vector<std::string>(val.begin(), val.end());
    }
    case AttrValueProto::ValueCase::kBools: {
      const auto& val = value_proto.bools().value();
      return std::vector<bool>(val.begin(), val.end());
    }
    case AttrValueProto::ValueCase::kInts: {
      const auto& val = value_proto.ints().value();
      return std::vector<std::int32_t>(val.begin(), val.end());
    }
    case AttrValueProto::ValueCase::kLongs: {
      const auto& val = value_proto.longs().value();
      return std::vector<std::int64_t>(val.begin(), val.end());
    }
    case AttrValueProto::ValueCase::kFloats: {
      const auto& val = value_proto.floats().value();
      return std::vector<float>(val.begin(), val.end());
    }
    case AttrValueProto::ValueCase::kDoubles: {
      const auto& val = value_proto.doubles().value();
      return std::vector<double>(val.begin(), val.end());
    }
    default:
      PADDLE_THROW(platform::errors::Unavailable("Invalid attribute type %d.",
                                                 value_proto.value_case()));
  }
}

}  // namespace note
}  // namespace piano
}  // namespace paddle
