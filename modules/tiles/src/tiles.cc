#include "motis/tiles/tiles.h"
#include <cstdlib>

#include "lmdb/lmdb.hpp"

#include "net/web_server/url_decode.h"

#include "tiles/db/pack_file.h"
#include "tiles/db/tile_database.h"
#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "utl/verify.h"

#include "motis/tiles/error.h"

#include "pbf_sdf_fonts_res.h"

namespace mm = motis::module;
namespace fb = flatbuffers;

namespace motis::tiles {

struct tiles::data {
  data(std::string const& path)
      : db_env_{::tiles::make_tile_database(path.c_str())},
        db_handle_{db_env_},
        render_ctx_{::tiles::make_render_ctx(db_handle_)},
        pack_handle_{path.c_str()} {}

  lmdb::env db_env_;
  ::tiles::tile_db_handle db_handle_;
  ::tiles::render_ctx render_ctx_;
  ::tiles::pack_handle pack_handle_;
};

tiles::tiles() : mm::module("Tiles", "tiles") {
  param(database_path_, "db", "/path/to/tiles.mdb");
}

tiles::~tiles() = default;

void tiles::init(mm::registry& reg) {
  if (!database_path_.empty()) {
    data_ = std::make_unique<data>(database_path_);
  }

  reg.register_op("/tiles", [&](auto const& msg) {
    if (!data_) {
      throw std::system_error(error::database_not_available);
    }

    auto tile =
        ::tiles::parse_tile_url(msg->get()->destination()->target()->str());
    if (!tile) {
      throw std::system_error(error::invalid_request);
    }

    ::tiles::null_perf_counter pc;
    auto rendered_tile = ::tiles::get_tile(
        data_->db_handle_, data_->pack_handle_, data_->render_ctx_, *tile, pc);

    mm::message_creator mc;
    std::vector<fb::Offset<HTTPHeader>> headers;
    fb::Offset<fb::String> payload;
    if (rendered_tile) {
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Type"),
          mc.CreateString("application/vnd.mapbox-vector-tile")));
      headers.emplace_back(CreateHTTPHeader(
          mc, mc.CreateString("Content-Encoding"), mc.CreateString("deflate")));
      payload = mc.CreateString(rendered_tile->data(), rendered_tile->size());
    } else {
      payload = mc.CreateString("");
    }

    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(mc, HTTPStatus_OK, mc.CreateVector(headers), payload)
            .Union());

    return make_msg(mc);
  });

  reg.register_op("/tiles/glyphs", [&](auto const& msg) {
    std::string decoded;
    net::url_decode(msg->get()->destination()->target()->str(), decoded);
    auto const mem = pbf_sdf_fonts_res::get_resource(decoded.substr(14));

    mm::message_creator mc;
    mc.create_and_finish(
        MsgContent_HTTPResponse,
        CreateHTTPResponse(
            mc, HTTPStatus_OK,
            mc.CreateVector(std::vector<fb::Offset<HTTPHeader>>{}),
            mc.CreateString(reinterpret_cast<char const*>(mem.ptr_), mem.size_))
            .Union());
    return make_msg(mc);
  });
}

}  // namespace motis::tiles
