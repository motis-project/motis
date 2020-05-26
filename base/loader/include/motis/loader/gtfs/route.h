#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "cista/reflection/comparable.h"

#include "motis/loader/gtfs/agency.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct category {
  CISTA_COMPARABLE()
  std::string name_;
  unsigned output_rule_{2};
};

struct route {
  route(agency const* agency, std::string id, std::string short_name,
        std::string long_name, std::string route_desc, int type)
      : agency_(agency),
        id_(std::move(id)),
        short_name_(std::move(short_name)),
        long_name_(std::move(long_name)),
        desc_{std::move(route_desc)},
        type_(type) {}

  static std::map<unsigned, category> s_types_;

  std::optional<category> get_category() const;

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
