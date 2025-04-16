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

std::string xml_to_str(pugi::xml_document const& doc) {
  std::stringstream ss{};
  doc.save(ss, "  ", pugi::format_default, pugi::xml_encoding::encoding_latin1);
  return ss.str();
}

pugi::xml_document parse(std::string const& s) {
  auto doc = pugi::xml_document{};
  doc.load_string(s.c_str());
  return doc;
}

}  // namespace motis::vdv_rt