file(GLOB_RECURSE SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp" "*.cc")
set(EXCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/python_api/")
list(FILTER SOURCE_FILES EXCLUDE REGEX "${EXCLUDE_PATH}")

# pybind11
execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${command_output}")

find_package(pybind11 REQUIRED)
# find_package(Python3 REQUIRED COMPONENTS Development)
add_library(mindie_llm_manager SHARED ${SOURCE_FILES})

if ("$ENV{MINDIE_ENABLE_PROF}" MATCHES "1")
        target_link_libraries(mindie_llm_manager PUBLIC ms_service_profiler)
endif()

target_link_libraries(mindie_llm_manager
        PUBLIC
        mindie_llm_utils
        mindie_llm_manager_v2
        config_manager
        pybind11::embed
        mindie_protobuf
        mindie_executor
        mindie_engine
)

install(TARGETS mindie_llm_manager DESTINATION lib)
