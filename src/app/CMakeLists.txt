# Include more application dependencies here
include(cli11)

add_executable(meme_app main.cpp)
# Direct dependencies
target_link_libraries(meme_app PRIVATE
    CLI11::CLI11
	mesh_metrics::mesh_metrics
)

set_target_properties(meme_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/app")
