#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::multiple_starting_foothpaths {

static loader::loader_options dataset_opt{
    {"test/schedule/multiple_starting_footpaths"}, "20151124"};

}  // namespace test::schedule::multiple_starting_foothpaths
}  // namespace motis
