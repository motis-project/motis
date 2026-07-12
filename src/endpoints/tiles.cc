#include "motis/endpoints/tiles.h"

#include <string>

#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "motis/tiles_data.h"

namespace motis::ep {

// Note: map font glyphs (SDF PBFs) are no longer served from here. They are
// shipped as static assets in the UI web folder (ui/static/glyphs/...) and
// referenced via the style's `glyphs` URL, together with the sprites. This
// endpoint only serves the actual vector tiles, which are the only genuinely
// server-side (data-derived) part of the map.
net::reply tiles::operator()(net::route_request const& req, bool) const {
  auto const url = boost::url_view{req.target()};

  if (req[boost::beast::http::field::accept_encoding].find("gzip") ==
      std::string_view::npos) {
    return net::web_server::empty_res_t{
        boost::beast::http::status::not_implemented, req.version()};
  }

  auto const tile = ::tiles::parse_tile_url(url.path());
  if (!tile.has_value()) {
    return net::web_server::empty_res_t{boost::beast::http::status::not_found,
                                        req.version()};
  }

  auto pc = ::tiles::null_perf_counter{};
  auto const rendered_tile =
      ::tiles::get_tile(tiles_data_.db_handle_, tiles_data_.pack_handle_,
                        tiles_data_.render_ctx_, *tile, pc);

  auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                           req.version()};
  res.insert(boost::beast::http::field::content_type,
             "application/vnd.mapbox-vector-tile");
  res.insert(boost::beast::http::field::content_encoding, "gzip");
  res.body() = rendered_tile.value_or("");
  res.keep_alive(req.keep_alive());
  return res;
}

}  // namespace motis::ep
