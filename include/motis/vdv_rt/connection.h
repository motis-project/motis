#pragma once

#include "nigiri/rt/vdv/vdv_update.h"

#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

auto const kHeaders = std::map<std::string, std::string>{
    {"Content-Type", "text/xml"}, {"Accept", "text/xml"}};

struct connection {
  connection(rt_entry::vdv_rt, nigiri::timetable const&, nigiri::source_idx_t);
  connection(connection&&) noexcept;

  std::string make_fetch_req();

  rt_entry::vdv_rt cfg_;
  std::string client_status_path_;
  std::string data_ready_path_;
  std::string server_status_addr_;
  std::string subscription_addr_;
  std::string fetch_data_addr_;
  nigiri::rt::vdv::updater upd_;
  std::atomic<vdv_rt_time> start_{now()};
};

}  // namespace motis::vdv_rt