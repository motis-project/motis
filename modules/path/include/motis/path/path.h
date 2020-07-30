#pragma once

#include <string>
#include <utility>

#include "motis/module/module.h"

namespace motis::path {

struct path : public motis::module::module {
  path();
  ~path() override;

  path(path const&) = delete;
  path& operator=(path const&) = delete;

  path(path&&) = delete;
  path& operator=(path&&) = delete;

  void import(motis::module::registry&) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

private:
  motis::module::msg_ptr boxes() const;

  motis::module::msg_ptr by_station_seq(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr by_trip_id(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr by_trip_id_batch(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr by_tile_feature(motis::module::msg_ptr const&) const;

  motis::module::msg_ptr path_tiles(motis::module::msg_ptr const&) const;

  std::vector<std::string> use_cache_;
  bool import_successful_{false};
};

}  // namespace motis::path
