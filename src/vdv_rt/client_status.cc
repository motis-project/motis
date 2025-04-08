#include "motis/vdv_rt/client_status.h"

#include "fmt/format.h"

#include "motis/vdv_rt/connection.h"
#include "motis/vdv_rt/time.h"
#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {

std::string client_status::operator()(std::string_view) const {
  auto doc = make_xml_doc();
  auto client_status_antwort_node = doc.append_child("ClientStatusAntwort");
  add_status_node(client_status_antwort_node);
  add_start_time_node(client_status_antwort_node, con_->start_);
  return xml_to_str(doc);
}

}  // namespace motis::vdv_rt