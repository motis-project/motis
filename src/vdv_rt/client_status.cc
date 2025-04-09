#include "motis/vdv_rt/client_status.h"

#include "motis/vdv_rt/connection.h"
#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {

std::string client_status::operator()(std::string_view) const {
  auto doc = make_xml_doc();
  auto client_status_res_node = doc.append_child("ClientStatusAntwort");

  auto status_node = client_status_res_node.append_child("Status");
  status_node.append_attribute("Zst") = timestamp(now()).c_str();
  status_node.append_attribute("Ergebnis") = "ok";

  auto start_time_node = client_status_res_node.append_child("StartDienstZst");
  start_time_node.append_child(pugi::node_pcdata)
      .set_value(timestamp(con_->start_).c_str());

  return xml_to_str(doc);
}

}  // namespace motis::vdv_rt