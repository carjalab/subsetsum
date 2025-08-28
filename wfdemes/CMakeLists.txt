cmake_minimum_required(VERSION 3.16...4.0)

add_library(WFDemesLib)
target_sources(WFDemesLib
    PRIVATE
        ./src/wrightfisher/Ensemble.cc
        ./src/wrightfisher/Parameters.cc
        ./src/wrightfisher/Simulation.cc)
target_include_directories(WFDemesLib
    PUBLIC
        ./include/wfdemes
    PRIVATE
        ./src)

# dependencies
target_link_libraries(WFDemesLib PRIVATE Boost::numeric_ublas)
target_link_libraries(WFDemesLib PRIVATE Boost::random)
target_link_libraries(WFDemesLib PRIVATE fmt::fmt)
target_link_libraries(WFDemesLib PRIVATE magic_enum::magic_enum)

# library requires at least C++17
target_compile_features(WFDemesLib PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
#source_group(
#  TREE "${PROJECT_SOURCE_DIR}/include"
#  PREFIX "Header Files"
#  FILES ${HEADER_LIST})

## library include directories (for IDEs)
#file(GLOB INCLUDE_LIST CONFIGURE_DEPENDS
#    "${PROJECT_SOURCE_DIR}/")
#
## library header files (for IDEs)
#file(GLOB HEADER_LIST CONFIGURE_DEPENDS
#    "${PROJECT_SOURCE_DIR}/include/wfdemes/*.hh"
#    "${PROJECT_SOURCE_DIR}/include/wfdemes/basemodel/*.hh"
#    "${PROJECT_SOURCE_DIR}/include/wfdemes/wrightfisher/*.hh")
#
## library source files
#file(GLOB SOURCE_LIST CONFIGURE_DEPENDS
#    "${PROJECT_SOURCE_DIR}/src/rngfactory.hh"
#    "${PROJECT_SOURCE_DIR}/src/wrightfisher/*.cc")


