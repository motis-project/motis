#pragma once

#include <map>
#include <memory>
#include <string>

#include "motis/loader/loaded_file.h"

namespace motis::loader::gtfs {

struct agency {
  agency(std::string id, std::string name, std::string timezone)
      : id_(std::move(id)),
        name_(std::move(name)),
        timezone_(std::move(timezone)) {}

  std::string id_;
  std::string name_;
  std::string timezone_;
};

using agency_map = std::map<std::string, std::unique_ptr<agency>>;

agency_map read_agencies(loaded_file);

}  // namespace motis::loader::gtfs
