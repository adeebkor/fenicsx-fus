cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME test_spec_op3d)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Set standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set flags
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/forms.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/forms.py -o
          ${CMAKE_CURRENT_SOURCE_DIR}/
  VERBATIM
  DEPENDS forms.py
  COMMENT "Compile forms.py using FFCx")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories("../../common" ".")

add_executable(${PROJECT_NAME} forms.c main.cpp)
target_link_libraries(${PROJECT_NAME} dolfinx)
