#include "motis/vdv_rt/connection.h"

namespace motis::vdv_rt {

connection::connection(config::timetable::dataset::vdv_rt const& cfg,
                       nigiri::timetable const& tt,
                       nigiri::source_idx_t const src)
    : cfg_{cfg},
      client_status_path_{
          fmt::format("/{}/aus/clientstatus.xml", cfg.server_name_)},
      data_ready_path_{
          fmt::format("/{}/aus/datenbereit.xml", cfg.server_name_)},
      server_status_addr_{fmt::format(
          "{}/{}/aus/status.xml", cfg.server_url_, cfg.client_name_)},
      subscription_addr_{fmt::format(
          "{}/{}/aus/aboverwalten.xml", cfg.server_url_, cfg.client_name_)},
      fetch_data_addr_{fmt::format(
          "{}/{}/aus/datenabrufen.xml", cfg.server_url_, cfg.client_name_)},
      upd_{tt, src} {}

connection::connection(connection&& other)
    : cfg_{std::move(other.cfg_)},
      client_status_path_{std::move(other.client_status_path_)},
      data_ready_path_{std::move(other.data_ready_path_)},
      server_status_addr_{std::move(other.server_status_addr_)},
      subscription_addr_{std::move(other.subscription_addr_)},
      fetch_data_addr_{std::move(other.fetch_data_addr_)},
      upd_{std::move(other.upd_)} {}

}  // namespace motis::vdv_rt