cmake_minimum_required(VERSION 3.21)
project(motis)

include_directories(include)

set(motis-footpaths-dbschema-dir ${CMAKE_CURRENT_SOURCE_DIR}/dbschema)
set(motis-footpaths-generated-headers-dir "${CMAKE_BINARY_DIR}/generated/motis/footpaths/dbschema")
file(GLOB_RECURSE motis-footpaths-dbschema-files ${motis-footpaths-dbschema-dir}/*.fbs)

build_flatbuffers(
  32                                            # num_bits
  "--everything-required"                       # commandline_options
  "${motis-footpaths-dbschema-files}"           # flatbuffers_schemas
  "${motis-footpaths-dbschema-dir}"             # schema_include_dirs
  generated-motis-footpaths-dbschema-headers    # custom_target_name
  ""                                            # additional_dependencies
  "${motis-footpaths-generated-headers-dir}"    # generated_includes_dir
  ""                                            # binary_schemas_dir
  ""                                            # copy_text_schemas_dir
)

file(GLOB_RECURSE motis-footpaths-files src/*.cc)
add_library(motis-footpaths STATIC ${motis-footpaths-files})
target_include_directories(motis-footpaths PUBLIC include)
target_include_directories(motis-footpaths PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../ppr/include)
target_compile_features(motis-footpaths PRIVATE cxx_std_20)
target_link_libraries(motis-footpaths
        boost-system
        boost-thread
        motis-module
        lmdb
        nigiri
        osmium
        geo
        ppr-routing
        )
target_compile_options(motis-footpaths PRIVATE ${MOTIS_CXX_FLAGS})
