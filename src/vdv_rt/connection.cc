#include "motis/vdv_rt/connection.h"

namespace motis::vdv_rt {

connection::connection(
    const motis::config::timetable::dataset::vdv_rt& vdv_rt_cfg)
    : client_status_path_{fmt::format("/{}/aus/clientstatus.xml",
                                      vdv_rt_cfg.server_name_)},
      data_ready_path_{
          fmt::format("/{}/aus/datenbereit.xml", vdv_rt_cfg.server_name_)},
      server_status_addr_{fmt::format("{}/{}/aus/status.xml",
                                      vdv_rt_cfg.server_url_,
                                      vdv_rt_cfg.client_name_)},
      subscription_addr_{fmt::format("{}/{}/aus/aboverwalten.xml",
                                     vdv_rt_cfg.server_url_,
                                     vdv_rt_cfg.client_name_)},
      fetch_data_addr_{fmt::format("{}/{}/aus/datenabrufen.xml",
                                   vdv_rt_cfg.server_url_,
                                   vdv_rt_cfg.client_name_)},
      start_{std::chrono::round<std::chrono::seconds>(
          std::chrono::system_clock::now())} {}

}  // namespace motis::vdv_rt