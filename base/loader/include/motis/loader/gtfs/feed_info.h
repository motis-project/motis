#pragma once

#include <map>
#include <string>

#include "motis/loader/loaded_file.h"

namespace motis {
namespace loader {
namespace gtfs {

struct feed {
  feed(std::string publisher_name, std::string version)
      : publisher_name_(std::move(publisher_name)),
        version_(std::move(version)) {}

  std::string publisher_name_;
  std::string version_;
};

using feed_map = std::map<std::string, feed>;

feed_map read_feed_publisher(loaded_file);

}  // namespace gtfs
}  // namespace loader
}  // namespace motis