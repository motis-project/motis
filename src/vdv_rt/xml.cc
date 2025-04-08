#include "motis/vdv_rt/xml.h"

#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

pugi::xml_document make_xml_doc() {
  auto doc = pugi::xml_document{};
  auto decl = doc.prepend_child(pugi::node_declaration);
  decl.append_attribute("version") = "1.0";
  decl.append_attribute("encoding") = "iso-8859-1";
  return doc;
}

void add_status_node(pugi::xml_node& node) {
  auto status_node = node.append_child("Status");
  status_node.append_attribute("Zst") = timestamp(now()).c_str();
  status_node.append_attribute("Ergebnis") = "ok";
}

void add_start_time_node(pugi::xml_node& node, sys_time const start) {
  auto start_dienst_zst_node = node.append_child("StartDienstZst");
  start_dienst_zst_node.append_child(pugi::node_pcdata)
      .set_value(timestamp(start).c_str());
}

std::string xml_to_str(pugi::xml_document const& doc) {
  std::stringstream ss{};
  doc.save(ss, "  ", pugi::format_default, pugi::xml_encoding::encoding_latin1);
  return ss.str();
}

}  // namespace motis::vdv_rt