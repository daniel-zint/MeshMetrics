# ###############################################################################
# WildMeshingToolkit library
# ###############################################################################

# Core library
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "*.cpp"
    "*.h"
    "*.hpp"
)
message(DEBUG "SRC_FILES: ${SRC_FILES}")

target_sources(mesh_metrics PRIVATE ${SRC_FILES})