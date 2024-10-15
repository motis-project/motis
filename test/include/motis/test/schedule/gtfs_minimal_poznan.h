#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis::test::schedule::gtfs_minimal_poznan {

static const loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/gtfs_minimal_poznan"},
    .schedule_begin_ = "20211217",
    .num_days_ = 2,
    .apply_rules_ = false};

}  // namespace motis::test::schedule::gtfs_minimal_poznan
