#pragma once

#include <memory>
#include <string>
#include <vector>

#include "motis/bikesharing/database.h"

namespace motis::bikesharing {

struct close_terminal {
  close_terminal(std::string id, double distance)
      : id_(std::move(id)), distance_(distance) {}

  std::string id_;
  double distance_;
};

struct geo_index {
  explicit geo_index(database const&);
  ~geo_index();

  geo_index(geo_index const&) = delete;
  geo_index& operator=(geo_index const&) = delete;

  geo_index(geo_index&&) = default;
  geo_index& operator=(geo_index&&) = default;

  std::vector<close_terminal> get_terminals(double lat, double lng,
                                            double radius) const;

private:
  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::bikesharing
