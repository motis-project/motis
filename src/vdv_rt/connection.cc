#include "motis/vdv_rt/connection.h"

#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {

connection::connection(rt_entry::vdv_rt cfg,
                       nigiri::timetable const& tt,
                       nigiri::source_idx_t const src)
    : cfg_{cfg},
      client_status_path_{
          fmt::format("/{}/aus/clientstatus.xml", cfg.server_name_)},
      data_ready_path_{
          fmt::format("/{}/aus/datenbereit.xml", cfg.server_name_)},
      server_status_addr_{
          fmt::format("{}/{}/aus/status.xml", cfg.url_, cfg.client_name_)},
      subscription_addr_{fmt::format(
          "{}/{}/aus/aboverwalten.xml", cfg.url_, cfg.client_name_)},
      fetch_data_addr_{fmt::format(
          "{}/{}/aus/datenabrufen.xml", cfg.url_, cfg.client_name_)},
      upd_{tt, src} {}

connection::connection(connection&& other) noexcept
    : cfg_{other.cfg_},
      client_status_path_{std::move(other.client_status_path_)},
      data_ready_path_{std::move(other.data_ready_path_)},
      server_status_addr_{std::move(other.server_status_addr_)},
      subscription_addr_{std::move(other.subscription_addr_)},
      fetch_data_addr_{std::move(other.fetch_data_addr_)},
      upd_{std::move(other.upd_)} {}

std::string connection::make_fetch_req() {
  auto doc = make_xml_doc();
  auto fetch_data_node = doc.append_child("DatenAbrufenAnfrage");
  fetch_data_node.append_attribute("Sender") = cfg_.client_name_.c_str();
  fetch_data_node.append_attribute("Zst") = timestamp(now()).c_str();
  auto all_datasets_node = fetch_data_node.append_child("DatensatzAlle");
  all_datasets_node.append_child(pugi::node_pcdata).set_value("true");
  return xml_to_str(doc);
}

}  // namespace motis::vdv_rt