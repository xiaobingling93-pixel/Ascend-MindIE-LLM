file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(foundation SHARED ${SOURCE_FILES})

target_link_libraries(foundation
    system_log
)

install(TARGETS foundation DESTINATION lib)
