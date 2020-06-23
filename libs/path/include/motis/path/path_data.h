#pragma once

#include <memory>

#include "tiles/get_tile.h"

#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/path/path_database.h"
#include "motis/path/path_index.h"

namespace motis::path {

struct path_database;
struct path_index;

constexpr auto const PATH_DATA_KEY = "path_data";

struct path_data {
  size_t trip_to_index(schedule const&, trip const*) const;

  module::msg_ptr get_response(size_t index, int zoom_level = -1) const;

  flatbuffers::Offset<PathSeqResponse> reconstruct_sequence(
      module::message_creator& mc, size_t index, int zoom_level = -1) const;

  std::unique_ptr<path_database> db_;
  std::unique_ptr<path_index> index_;

  tiles::render_ctx render_ctx_;
};

}  // namespace motis::path
