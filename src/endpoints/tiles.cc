#include "motis/endpoints/tiles.h"

#include <string>

#include "net/web_server/url_decode.h"

#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "motis/tiles_data.h"

#include "pbf_sdf_fonts_res.h"

using namespace std::string_view_literals;

namespace motis::ep {

net::reply tiles::operator()(net::route_request const& req, bool) const {
  auto const url = boost::url_view{req.target()};
  if (url.path().starts_with("/tiles/glyphs")) {
    std::string decoded;
    net::url_decode(url.path(), decoded);

    // Rewrite old font name "Noto Sans Display Regular" to "Noto Sans Regular".
    constexpr auto kDisplay = " Display"sv;
    auto res_name = decoded.substr(14);
    if (auto const display_pos = res_name.find(kDisplay);
        display_pos != std::string::npos) {
      res_name.erase(display_pos, kDisplay.length());
    }

    try {
      auto const mem = pbf_sdf_fonts_res::get_resource(res_name);
      auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                               req.version()};
      res.body() =
          std::string_view{reinterpret_cast<char const*>(mem.ptr_), mem.size_};
      res.keep_alive(req.keep_alive());
      return res;
    } catch (std::out_of_range const&) {
      throw net::not_found_exception{res_name};
    }
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
