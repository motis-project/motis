#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis::test::schedule::gtfs_minimal_swiss {

static const loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/gtfs_minimal_swiss"},
    .schedule_begin_ = "20190625",
    .num_days_ = 6,
    .apply_rules_ = false};

}  // namespace motis::test::schedule::gtfs_minimal_swiss
