cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME print_geometric_data)
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

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories("../../common" ".")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} dolfinx)
