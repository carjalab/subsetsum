cmake_minimum_required(VERSION 3.16...4.0)

project(WFDemes
    VERSION 0.1
    DESCRIPTION "Wright-Fisher model on demes."
    LANGUAGES CXX)

# check that this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # project-wide settings
    include(CTest)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED On)
    set(CMAKE_CXX_EXTENSIONS OFF)  # `-std=c++xx`, not `g++xx`
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
endif()

# find dependencies
find_package(Boost 1.88.0 REQUIRED COMPONENTS numeric_ublas random)
find_package(fmt CONFIG REQUIRED)
find_package(magic_enum CONFIG REQUIRED)

# library code
add_subdirectory(wfdemes)
# executable code
add_subdirectory(app)
