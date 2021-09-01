cmake_minimum_required(VERSION 3.14)
project(OscillatorVisIt)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(MPI REQUIRED COMPONENTS C CXX)
find_path(DIY_INCLUDE_DIR critical-resource.hpp PATH_SUFFIXES diy REQUIRED) 
find_path(VISIT_INSTALL_DIR PATH_SUFFIXES libsim) 


set(HEADER
Block.h
Oscillator.h
Particles.h
analysis.h
bridge.h
)

set(SOURCES
main.cpp
Block.cpp
Oscillator.cpp
Particles.cpp
analysis.cpp
bridge.cpp
)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS}")
set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_definitions(-Werror=return-type)
endif()

add_executable(oscillator ${HEADER} ${SOURCES})
target_include_directories(oscillator PRIVATE ${DIY_INCLUDE_DIR} ${VISIT_INSTALL_DIR}/libsim/V2/include)
message(MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES})
target_link_libraries(oscillator PRIVATE ${MPI_CXX_LIBRARIES} ${VISIT_INSTALL_DIR}/libsim/V2/lib/libsimV2dyn.so stdc++fs dl)
target_compile_definitions(oscillator PRIVATE VISIT_INSTALL_DIR="${VISIT_INSTALL_DIR}")
target_compile_features(oscillator PRIVATE cxx_std_17)
