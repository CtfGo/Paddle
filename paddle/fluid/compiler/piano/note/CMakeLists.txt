cc_library(note_opcode SRCS opcode.cc DEPS errors enforce)
cc_test(note_opcode_test SRCS opcode_test.cc DEPS note_opcode)

proto_library(note_proto SRCS note.proto)
target_compile_options(note_proto PUBLIC "-Wno-extra")

cc_library(note_ir SRCS instruction.cc function.cc module.cc DEPS note_opcode note_proto piano_data_description)
cc_test(note_ir_test SRCS note_ir_test.cc DEPS note_ir)

cc_library(note_template_util SRCS element_type_util.cc populate_attribute_value.cc DEPS note_proto)
cc_test(note_element_type_util_test SRCS element_type_util_test.cc DEPS note_template_util)
cc_test(note_populate_attribute_value_test SRCS populate_attribute_value_test.cc DEPS note_template_util)
