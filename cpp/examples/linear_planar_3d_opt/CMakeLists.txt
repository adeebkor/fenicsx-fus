cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME linear_planar3d_opt)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Ofast -march=native -mprefer-vector-width=512 ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native  -mprefer-vector-width=512 ${CMAKE_C_FLAGS} -Wall")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/forms.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/forms.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS forms.ufl
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)

include_directories("../../common" ".")

# Executable
add_executable(${PROJECT_NAME} forms.c main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

message(STATUS ${CMAKE_CXX_FLAGS})