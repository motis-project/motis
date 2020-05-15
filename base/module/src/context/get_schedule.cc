#include "motis/module/context/get_schedule.h"

#include "motis/core/schedule/schedule.h"

namespace motis::module {

schedule& get_schedule() {
  return *ctx::current_op<ctx_data>()
              ->data_.dispatcher_->shared_data_
              .get<schedule_data>(SCHEDULE_DATA_KEY)
              .schedule_;
}

}  // namespace motis::module