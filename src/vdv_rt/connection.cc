#include "motis/vdv_rt/connection.h"

namespace motis::vdv_rt {

connection::connection(const motis::config::vdv_rt& vdv_rt_cfg)
    : vdv_rt_cfg_{vdv_rt_cfg},
      client_status_path_{
          fmt::format("/{}/aus/clientstatus.xml", vdv_rt_cfg_.server_name_)},
      data_ready_path_{
          fmt::format("/{}/aus/datenbereit.xml", vdv_rt_cfg_.server_name_)},
      server_status_addr_{fmt::format("{}/{}/aus/status.xml",
                                      vdv_rt_cfg_.server_url_,
                                      vdv_rt_cfg_.client_name_)},
      manage_subscription_addr_{fmt::format("{}/{}/aus/aboverwalten.xml",
                                            vdv_rt_cfg_.server_url_,
                                            vdv_rt_cfg_.client_name_)},
      fetch_data_addr_{fmt::format("{}/{}/aus/datenabrufen.xml",
                                   vdv_rt_cfg_.server_url_,
                                   vdv_rt_cfg_.client_name_)} {}

}  // namespace motis::vdv_rt