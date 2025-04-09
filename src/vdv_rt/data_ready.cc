#include "motis/vdv_rt/data_ready.h"

#include "motis/vdv_rt/xml.h"

namespace motis::vdv_rt {
std::string data_ready::operator()(std::string_view) const {
  auto doc = make_xml_doc();
  auto data_ready_node = doc.append_child("DatenBereitAntwort");

  auto ack_node = data_ready_node.append_child("Bestaetigung");
  ack_node.append_attribute("Zst") = timestamp(now()).c_str();
  ack_node.append_attribute("Ergebnis") = "ok";
  ack_node.append_attribute("Fehlernummer") = "0";

  return xml_to_str(doc);
}

}  // namespace motis::vdv_rt