#pragma once

#include "nigiri/rt/vdv/vdv_update.h"

#include "motis/config.h"
#include "motis/fwd.h"
#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

struct connection {
  connection(config::timetable::dataset::vdv_rt const&,
             nigiri::timetable const&,
             nigiri::source_idx_t);
  connection(connection&&);

  std::string make_fetch_req();

  config::timetable::dataset::vdv_rt const& cfg_;
  std::string client_status_path_;
  std::string data_ready_path_;
  std::string server_status_addr_;
  std::string subscription_addr_;
  std::string fetch_data_addr_;
  nigiri::rt::vdv::updater upd_;
  std::atomic<vdv_rt_time> start_{now()};
};

}  // namespace motis::vdv_rt