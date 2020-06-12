#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::wzr_realtime {

static loader::loader_options dataset_opt{
    .dataset_ = {"test/schedule/wzr_realtime"},
    .schedule_begin_ = "20151124",
    .wzr_classes_path_ = "base/loader/test_resources/wzr/wzr_classes.csv",
    .wzr_matrix_path_ = "base/loader/test_resources/wzr/wzr_matrix.txt"};

static loader::loader_options dataset_opt_no_rules{
    .dataset_ = {"test/schedule/wzr_realtime"},
    .schedule_begin_ = "20151124",
    .apply_rules_ = false};

}  // namespace test::schedule::wzr_realtime
}  // namespace motis
