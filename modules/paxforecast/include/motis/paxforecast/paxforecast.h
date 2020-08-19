#pragma once

#include <memory>
#include <string>
#include <vector>

#include "conf/date_time.h"

#include "motis/module/module.h"

namespace motis::paxforecast {

struct paxforecast : public motis::module::module {
  paxforecast();
  ~paxforecast() override;

  paxforecast(paxforecast const&) = delete;
  paxforecast& operator=(paxforecast const&) = delete;

  paxforecast(paxforecast&&) = delete;
  paxforecast& operator=(paxforecast&&) = delete;

  void init(motis::module::registry&) override;

private:
  void on_monitoring_event(motis::module::msg_ptr const& msg);
};

}  // namespace motis::paxforecast
