# Check required CMake version
set(REQUIRED_CMAKE_VERSION "3.24.0")

cmake_minimum_required(VERSION ${REQUIRED_CMAKE_VERSION})

cmake_policy(SET CMP0135 NEW)

# Enable ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    option(WMTK_WITH_CCACHE "Enable ccache when building WMTK" ${mesh_metrics_TOPLEVEL_PROJECT})
else()
    option(WMTK_WITH_CCACHE "Enable ccache when building WMTK" OFF)
endif()
if(WMTK_WITH_CCACHE AND CCACHE_PROGRAM)
    set(ccacheEnv
        CCACHE_BASEDIR=${CMAKE_BINARY_DIR}
        CCACHE_SLOPPINESS=clang_index_store,include_file_ctime,include_file_mtime,locale,pch_defines,time_macros
    )
    foreach(lang IN ITEMS C CXX)
        set(CMAKE_${lang}_COMPILER_LAUNCHER
            ${CMAKE_COMMAND} -E env ${ccacheEnv} ${CCACHE_PROGRAM}
        )
    endforeach()
endif()

# ###############################################################################
project(MeshMetrics DESCRIPTION "Evaluate Metrics on Triangle Meshes" LANGUAGES C CXX)

# ###############################################################################
option (BUILD_SHARED_LIBS "Build Shared Libraries" OFF) # we globally want to disable this option due to use of TBB

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Sort projects inside the solution
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Generate position independent code by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ###############################################################################
# TriMeshMetrics library
# ###############################################################################

# Dependencies
include(libigl)
include(pybind11)
include(mshio)

# Core library
add_library(mesh_metrics)
add_library(mesh_metrics::mesh_metrics ALIAS mesh_metrics)
set_property(TARGET mesh_metrics PROPERTY COMPILE_WARNING_AS_ERROR  ON)

add_subdirectory(src/lib/meme)

# Group source files for IDEs
file(GLOB_RECURSE MEME_FILES_FOR_SOURCE_GROUP "src/wmtk/*.cpp" "src/wmtk/*.h" "src/wmtk/*.hpp")
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/src/wmtk" PREFIX "wmtk" FILES ${MEME_FILES_FOR_SOURCE_GROUP})

target_include_directories(mesh_metrics PUBLIC src/lib)

# Compile definitions
target_compile_definitions(mesh_metrics PUBLIC _USE_MATH_DEFINES)
target_compile_definitions(mesh_metrics PUBLIC NOMINMAX)

# C++ standard
target_compile_features(mesh_metrics PUBLIC cxx_std_17)

target_link_libraries(mesh_metrics PUBLIC
    Eigen3::Eigen
    igl::core
    mshio::mshio
)

if(MSVC)
    target_compile_options(mesh_metrics PUBLIC /MP)
    target_compile_definitions(mesh_metrics PUBLIC _SILENCE_ALL_MS_EXT_DEPRECATION_WARNINGS)
endif()

# ###############################################################################
# Subdirectories
# ###############################################################################

add_subdirectory(src/app)
add_subdirectory(src/pymeme)


