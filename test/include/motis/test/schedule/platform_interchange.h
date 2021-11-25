#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::platform_interchange {

static loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/platform_interchange"},
    .schedule_begin_ = "20151124",
    .use_platforms_ = true};

static loader::loader_options dataset_without_platforms_opt{
    .dataset_ = {"test/schedule/platform_interchange"},
    .schedule_begin_ = "20151124",
    .use_platforms_ = false};

}  // namespace test::schedule::platform_interchange
}  // namespace motis
