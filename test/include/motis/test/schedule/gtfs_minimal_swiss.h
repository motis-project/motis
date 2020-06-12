#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::gtfs_minimal_swiss {

static loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/gtfs_minimal_swiss"},
    .schedule_begin_ = "20190625",
    .num_days_ = 6,
    .apply_rules_ = false};

}  // namespace test::schedule::gtfs_minimal_swiss
}  // namespace motis