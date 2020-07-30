#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::simple_realtime {

static loader::loader_options dataset_opt{{"test/schedule/simple_realtime"},
                                          "20151124"};

static loader::loader_options dataset_opt_short{
    .dataset_ = {"test/schedule/simple_realtime"},
    .schedule_begin_ = "20151124",
    .num_days_ = 1};

static loader::loader_options dataset_opt_long{
    .dataset_ = {"test/schedule/simple_realtime"},
    .schedule_begin_ = "20151124",
    .num_days_ = 6};

}  // namespace test::schedule::simple_realtime
}  // namespace motis
