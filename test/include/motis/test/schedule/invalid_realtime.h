#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::invalid_realtime {

static loader::loader_options dataset_opt{{"test/schedule/invalid_realtime"},
                                          "20151124"};

static loader::loader_options dataset_opt_no_rules{
    .dataset_ = {"test/schedule/invalid_realtime"},
    .schedule_begin_ = "20151124",
    .apply_rules_ = false};

}  // namespace test::schedule::invalid_realtime
}  // namespace motis
