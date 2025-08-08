#include "motis/endpoints/tiles.h"

#include <string>

#include "net/web_server/url_decode.h"

#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "motis/tiles_data.h"

#include "pbf_sdf_fonts_res.h"

namespace motis::ep {

net::reply tiles::operator()(net::route_request const& req, bool) const {
  auto const url = boost::url_view{req.target()};
  if (url.path().starts_with("/tiles/glyphs")) {
    std::string decoded;
    net::url_decode(url.path(), decoded);
    auto const mem = pbf_sdf_fonts_res::get_resource(decoded.substr(14));

    auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                             req.version()};
    res.body() =
        std::string_view{reinterpret_cast<char const*>(mem.ptr_), mem.size_};
    res.keep_alive(req.keep_alive());
    return res;
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
  res.insert(boost::beast::http::field::content_encoding, "deflate");
  res.body() = rendered_tile.value_or("");
  res.keep_alive(req.keep_alive());
  return res;
}

}  // namespace motis::ep
