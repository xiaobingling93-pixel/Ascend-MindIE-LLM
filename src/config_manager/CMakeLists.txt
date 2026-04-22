file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(config_manager SHARED ${SOURCE_FILES})

target_link_libraries(config_manager
        mindie_llm_utils
)

install(TARGETS config_manager DESTINATION lib)
