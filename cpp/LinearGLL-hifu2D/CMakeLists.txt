cmake_minimum_required(VERSION 3.16)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PROJECT_NAME hifu)

<<<<<<< HEAD
set(CMAKE_CXX_FLAGS "-Ofast -mprefer-vector-width=512 -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -mprefer-vector-width=512 -march=native  ${CMAKE_C_FLAGS} -Wall")
=======
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native  ${CMAKE_C_FLAGS} -Wall")
>>>>>>> 3c2105645a2178d06a12ac452d12cdcb13ddb906

project(${PROJECT_NAME})

# Get DOLFINx configuration data (DOLFINXConfig.cmake must be in
# DOLFINX_CMAKE_CONFIG_PATH)
if (NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# Executable
add_executable(${PROJECT_NAME} forms.c main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfinx)

message(STATUS ${CMAKE_CXX_FLAGS})