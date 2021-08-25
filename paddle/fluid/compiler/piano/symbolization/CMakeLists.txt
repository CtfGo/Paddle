cc_library(note_builder SRCS note_builder.cc DEPS string_helper note_opcode piano_data_description)
cc_test(note_builder_test SRCS note_builder_test.cc DEPS note_builder)

cc_library(piano_meta_op SRCS meta_op.cc shape_inference.cc DEPS string_helper note_opcode piano_data_description note_builder)
