#pragma once

#include "motis/config.h"
#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

struct connection {
  explicit connection(motis::config::vdv_rt const&);

  motis::config::vdv_rt const& vdv_rt_cfg_;
  std::string client_status_path_;
  std::string data_ready_path_;
  std::string server_status_addr_;
  std::string manage_subscription_addr_;
  std::string fetch_data_addr_;
  sys_time start_;
};

}  // namespace motis::vdv_rt