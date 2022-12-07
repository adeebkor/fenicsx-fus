cmake_minimum_required(VERSION 3.16)
project(hifu-simulation-tests)

project(${PROJECT_NAME} LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Dolfinx config file
find_package(DOLFINX REQUIRED)

add_custom_command(
    OUTPUT forms.c
    COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/forms.py -o ${CMAKE_CURRENT_SOURCE_DIR}
    VERBATIM DEPENDS forms.py COMMENT "Compile forms.py using FFCx")

# Make test executable
set(TEST_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/forms.c
  ${CMAKE_CURRENT_SOURCE_DIR}/operators.cpp)


include_directories("../common" ".")
add_executable(unittests ${TEST_SOURCES})
target_link_libraries(unittests PRIVATE dolfinx)
target_compile_options(unittests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Werror -Wextra -pedantic>)

# Enable testing
enable_testing()

# Test target
add_test(unittests unittests)