#pragma once

#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

#include "motis/rsl/output/output.h"
#include "motis/rsl/rsl_data.h"
#include "motis/rsl/statistics.h"
#include "motis/rsl/stats_writer.h"

namespace motis::paxassign {

struct paxassign : public motis::module::module {
  paxassign();
  ~paxassign() override;

  paxassign(paxassign const&) = delete;
  paxassign& operator=(paxassign const&) = delete;

  paxassign(paxassign&&) = delete;
  paxassign& operator=(paxassign&&) = delete;

  void init(motis::module::registry&) override;

private:
  void on_forecast(motis::module::msg_ptr const& msg);
};

}  // namespace motis::paxassign
