pybind11_add_module(pymeme pymeme.cpp)

target_link_libraries(pymeme PUBLIC
    mesh_metrics::mesh_metrics
    pybind11::module
)

set_target_properties(pymeme PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/app")