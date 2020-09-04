#pragma once

#include "motis/module/message.h"
#include "motis/loader/loader_options.h"

namespace motis {

struct schedule;

namespace test::schedule::platform_interchange {

static loader::loader_options dataset_opt{
    {"test/schedule/platform_interchange"}, "20151124"};

// TODO(pablo): NYI
static loader::loader_options dataset_without_platforms_opt{
    {"test/schedule/platform_interchange"}, "20151124"};

}  // namespace test::schedule::platform_interchange
}  // namespace motis
