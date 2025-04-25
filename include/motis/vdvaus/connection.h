#pragma once

#include "nigiri/rt/vdv/vdv_update.h"

#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/vdvaus/time.h"

namespace motis::vdvaus {

auto const kHeaders = std::map<std::string, std::string>{
    {"Content-Type", "text/xml"}, {"Accept", "text/xml"}};

struct connection {
  connection(rt_ep_config::vdvaus,
             nigiri::timetable const&,
             nigiri::source_idx_t);
  connection(connection&&) noexcept;

  std::string make_fetch_req();

  rt_ep_config::vdvaus cfg_;
  std::string client_status_path_;
  std::string data_ready_path_;
  std::string server_status_addr_;
  std::string subscription_addr_;
  std::string fetch_data_addr_;
  nigiri::rt::vdv::updater upd_;
  std::atomic<vdvaus_time> start_{now()};
};

}  // namespace motis::vdvaus