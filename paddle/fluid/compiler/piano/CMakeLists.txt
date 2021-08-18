add_subdirectory(backends)
add_subdirectory(note)

cc_library(piano_data_description SRCS layout.cc shape.cc DEPS string_helper note_proto)
cc_test(piano_layout_test SRCS layout_test.cc DEPS piano_data_description)
cc_test(piano_shape_test SRCS shape_test.cc DEPS piano_data_description)

cc_library(note_builder SRCS note_builder.cc DEPS string_helper note_opcode piano_data_description)
cc_test(note_builder_test SRCS note_builder_test.cc DEPS note_builder)
