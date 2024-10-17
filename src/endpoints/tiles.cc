#include "motis/endpoints/tiles.h"

#include <string>

#include "net/web_server/url_decode.h"

#include "tiles/get_tile.h"
#include "tiles/parse_tile_url.h"
#include "tiles/perf_counter.h"

#include "motis/tiles_data.h"

#include "pbf_sdf_fonts_res.h"

namespace motis::ep {

http_response tiles::operator()(boost::urls::url_view const& url) const {
  if (url.path().starts_with("/tiles/glyphs")) {
    auto const mem = pbf_sdf_fonts_res::get_resource(url.path().substr(14));

    auto res = http_response{boost::beast::http::status::ok, 11};
    res.body() =
        std::string_view{reinterpret_cast<char const*>(mem.ptr_), mem.size_};
    return res;
  }

  auto const tile = ::tiles::parse_tile_url(url.path());

  auto pc = ::tiles::null_perf_counter{};
  auto const rendered_tile =
      ::tiles::get_tile(tiles_data_.db_handle_, tiles_data_.pack_handle_,
                        tiles_data_.render_ctx_, *tile, pc);

  auto res = http_response{boost::beast::http::status::ok, 11};
  res.insert(boost::beast::http::field::content_type,
             "application/vnd.mapbox-vector-tile");
  res.insert(boost::beast::http::field::content_encoding, "deflate");
  res.body() = rendered_tile.value_or("");
  return res;
}

}  // namespace motis::ep