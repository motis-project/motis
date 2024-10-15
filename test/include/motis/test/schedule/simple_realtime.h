#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::simple_realtime {

static auto const dataset_opt =
    loader::loader_options{.dataset_ = {"test/schedule/simple_realtime"},
                           .schedule_begin_ = "20151124"};

static auto const dataset_opt_short =
    loader::loader_options{.dataset_ = {"test/schedule/simple_realtime"},
                           .schedule_begin_ = "20151124",
                           .num_days_ = 1};

static auto const dataset_opt_long =
    loader::loader_options{.dataset_ = {"test/schedule/simple_realtime"},
                           .schedule_begin_ = "20151124",
                           .num_days_ = 6};

}  // namespace test::schedule::simple_realtime
}  // namespace motis
