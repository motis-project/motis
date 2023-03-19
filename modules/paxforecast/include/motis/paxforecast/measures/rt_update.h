#pragma once

#include <string>

#include "motis/core/schedule/time.h"

#include "motis/protocol/RISContentType_generated.h"

#include "motis/paxforecast/measures/recipients.h"

namespace motis::paxforecast::measures {

struct rt_update {
  recipients recipients_;
  time time_{};
  motis::ris::RISContentType type_{};
  std::string content_;
};

}  // namespace motis::paxforecast::measures
