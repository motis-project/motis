#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "motis/loader/gtfs/agency.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

class route {
public:
  route(agency const* agency, std::string id, std::string short_name,
        std::string long_name, std::string route_desc, int type)
      : agency_(agency),
        id_(std::move(id)),
        short_name_(std::move(short_name)),
        long_name_(std::move(long_name)),
        desc_{std::move(route_desc)},
        type_(type) {}

  static std::map<unsigned, std::string> s_types_;

  std::optional<std::string> category() const;

  agency const* agency_;
  std::string id_;
  std::string short_name_;
  std::string long_name_;
  std::string desc_;
  int type_;
};

using route_map = std::map<std::string, std::unique_ptr<route>>;

route_map read_routes(loaded_file, agency_map const&);

}  // namespace motis::loader::gtfs
