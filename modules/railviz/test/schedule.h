#pragma once

#include "motis/loader/loader_options.h"

namespace motis::railviz {

static loader::loader_options dataset_opt{
    .dataset_ = {"modules/railviz/test_resources/schedule"},
    .schedule_begin_ = "20151121"};

}  // namespace motis::railviz
