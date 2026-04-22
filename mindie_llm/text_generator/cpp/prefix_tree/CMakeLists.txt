file(GLOB SOURCE_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")

add_library(prefix_tree SHARED ${SOURCE_FILES})
set_target_properties(prefix_tree PROPERTIES PREFIX "_")

target_link_libraries(prefix_tree PUBLIC stdc++)

install(TARGETS prefix_tree DESTINATION lib)
