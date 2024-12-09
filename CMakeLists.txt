cmake_minimum_required(VERSION 3.12)

project(DifferentiableOwen)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

option(WITH_GEOGRAM "Build with geogram" OFF)
option(WITH_UTK "Build with UTK" OFF)

include(cli11)

if (${WITH_UTK})
    include(utk)
ENDIF()

ADD_LIBRARY(SOURCES
    # Utils
    src/utils/Array.cpp
    src/utils/BinaryUtils.cpp
    src/utils/ProgressBar.cpp
    src/utils/Logger.cpp

    # Losses
    src/loss/fourier/PCF.cpp
    src/loss/semi_discrete_ot_2d/transport.cpp
    src/loss/semi_discrete_ot_3d/geogram_wrapper.cpp
    src/loss/gbn/gbn.cpp
    src/loss/integration/Integrands.cpp
    src/loss/loss.cpp

    # Owen
    src/SmoothOwen/SmoothOwen.cpp
    # src/SmoothOwen/SmoothPermutationOwen.cpp
)
add_compile_definitions(DATA_PATH="${CMAKE_SOURCE_DIR}/data")


target_include_directories(SOURCES PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(SOURCES PRIVATE "/openmp")


function(create_exe exename mainname)
    add_executable(${exename} src/main/main_${mainname}.cpp)
    target_include_directories(${exename} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${exename} SOURCES CLI11::CLI11)
    target_compile_options(${exename} PRIVATE "/openmp")
endfunction()

create_exe(DifferentiableOwenW2 w2)
create_exe(DifferentiableOwenGBN gbn)
create_exe(DifferentiableOwenInt int)
create_exe(DifferentiableOwenpcf pcf)

create_exe(DifferentiableOwenProgressiveW2 progressive_w2)
create_exe(DifferentiableOwenProgressiveGBN progressive_gbn)
create_exe(DifferentiableOwenProgressiveInt progressive_int)