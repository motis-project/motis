#pragma once

#include <string>
#include <utility>

#include "pugixml.hpp"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris::risml {

flatbuffers::Offset<flatbuffers::String> inline parse_station(
    flatbuffers::FlatBufferBuilder& b, pugi::xml_node const& node,
    char const* eva_attr_name) {
  auto const& attr = node.attribute(eva_attr_name);
  if (!attr.empty()) {
    std::string eva_string(attr.value());
    if (eva_string.size() == 6) {
      eva_string.insert(0, 1, '0');
    }
    return b.CreateString(eva_string);
  }
  return b.CreateString("");
}

flatbuffers::Offset<flatbuffers::String> inline parse_station(
    flatbuffers::FlatBufferBuilder& fbb, pugi::xml_node const& e_node) {
  auto const& station_node = e_node.child("Bf");
  return parse_station(fbb, station_node, "EvaNr");
}

}  // namespace motis::ris::risml
