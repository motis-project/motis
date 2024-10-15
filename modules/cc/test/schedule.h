#pragma once

#include "motis/loader/loader_options.h"

namespace motis::cc {

static loader::loader_options dataset_opt{
    .dataset_ = {"modules/cc/test_resources/schedule"},
    .schedule_begin_ = "20151124"};

}  // namespace motis::cc
