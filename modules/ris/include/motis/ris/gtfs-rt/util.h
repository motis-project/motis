#pragma once

#include <string>

namespace motis::ris::gtfsrt {

std::string json_to_protobuf(std::string const& json);
std::string protobuf_to_json(std::string const& protobuf);

}  // namespace motis::ris::gtfsrt