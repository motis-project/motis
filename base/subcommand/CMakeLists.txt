project(motis)
cmake_minimum_required(VERSION 3.19)

add_library(motis-subcommand src/subcommand.cc)
target_include_directories(motis-subcommand PUBLIC include)