cmake_minimum_required(VERSION 3.16...4.0)
add_executable(WFDemesApp)
target_sources(WFDemesApp
    PRIVATE
        ./src/main.cc)
target_include_directories(WFDemesApp PRIVATE ./src)
target_link_libraries(WFDemesApp PRIVATE WFDemesLib)
target_link_libraries(WFDemesApp PRIVATE fmt::fmt)
target_compile_features(WFDemesApp PRIVATE cxx_std_17)

# application source files
#file(GLOB SOURCE_LIST CONFIGURE_DEPENDS
#    "${PROJECT_SOURCE_DIR}/src/formatting.hh"
#    "${PROJECT_SOURCE_DIR}/src/parsing.hh"
#    "${PROJECT_SOURCE_DIR}/src/usagehelp.hh")
# target_include_directories(wfdemes PRIVATE ../src)
