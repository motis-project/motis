#pragma once

#include "ctx/ctx.h"

#include "motis/core/schedule/schedule_data_key.h"
#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"

namespace motis {

struct schedule;

namespace module {

schedule& get_schedule();

}  // namespace module
}  // namespace motis
