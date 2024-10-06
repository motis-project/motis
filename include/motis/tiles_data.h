#pragma once

#include <string>

#include "tiles/db/tile_database.h"
#include "tiles/get_tile.h"

namespace motis {

struct tiles_data {
  tiles_data(std::string const& path, std::size_t const db_size)
      : db_env_{::tiles::make_tile_database(path.c_str(), db_size)},
        db_handle_{db_env_},
        render_ctx_{::tiles::make_render_ctx(db_handle_)},
        pack_handle_{path.c_str()} {}

  lmdb::env db_env_;
  ::tiles::tile_db_handle db_handle_;
  ::tiles::render_ctx render_ctx_;
  ::tiles::pack_handle pack_handle_;
};

}  // namespace motis