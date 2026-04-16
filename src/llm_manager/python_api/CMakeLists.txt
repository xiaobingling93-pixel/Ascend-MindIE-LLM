file(GLOB LLM_MANAGER_STATIC_SRCS
        ${PROJECT_SOURCE_DIR}/src/include/llm_manager/*.h
        ${CMAKE_CURRENT_LIST_DIR}/*.cpp
        ${PROJECT_SOURCE_DIR}/src/llm_manager/*.cpp
        )
include_directories(
        ${pybind11_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/llm_manager
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../../utils/include/mindie_llm
        ${THIRD_PARTY_OUTPUT_DIR}/nlohmann/include
)

execute_process(
    COMMAND bash -c "python3 -m pybind11 --cmakedir"
    OUTPUT_VARIABLE command_output
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR "${command_output}")
find_package(pybind11 REQUIRED)

add_library(llm_manager_python SHARED ${LLM_MANAGER_STATIC_SRCS})
set_target_properties(llm_manager_python PROPERTIES PREFIX "")
target_link_libraries(llm_manager_python
        PUBLIC
        -Wl,--start-group
        pybind11::embed
        -Wl,--end-group
        mindie_llm_manager
)

install(TARGETS llm_manager_python DESTINATION lib)
