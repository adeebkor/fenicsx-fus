cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME mwe_parallel_eval)
project(${PROJECT_NAME})

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-O3 -march=native  ${CMAKE_C_FLAGS} -Wall")

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/form.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/form.ufl -o ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS form.ufl
)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

find_package(DOLFINX REQUIRED)

include_directories("../../common" ".")

# Executable
add_executable(${PROJECT_NAME} form.c main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx blas lapack)

message(STATUS ${CMAKE_CXX_FLAGS})