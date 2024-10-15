#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::raptor_moc {

static const loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/raptor_max_occupancy"},
    .schedule_begin_ = "20210826",
    .num_days_ = 7,
    .apply_rules_ = false};

}  // namespace test::schedule::gtfs_minimal_swiss
}  // namespace motis
