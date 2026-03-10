#include "motis/endpoints/map/shapes_debug.h"

#include <charconv>
#include <set>
#include <string>
#include <string_view>
#include <tuple>

#include "boost/json.hpp"

#include "fmt/format.h"

#include "utl/to_vec.h"
#include "utl/verify.h"

#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "osr/routing/map_matching_debug.h"

#include "motis/route_shapes.h"
#include "motis/tag_lookup.h"

namespace motis::ep {

namespace {

std::uint64_t parse_route_idx(std::string_view const path) {
  auto const slash = path.find_last_of('/');
  auto const idx_str =
      slash == std::string_view::npos ? path : path.substr(slash + 1U);
  utl::verify(!idx_str.empty(), "missing route index");

  auto route_idx = std::uint64_t{};
  auto const* begin = idx_str.data();
  auto const* end = begin + idx_str.size();
  auto const [ptr, ec] = std::from_chars(begin, end, route_idx);
  utl::verify(ec == std::errc{} && ptr == end, "invalid route index '{}'",
              idx_str);
  return route_idx;
}

boost::json::object build_caller_data(nigiri::timetable const& tt,
                                      tag_lookup const& tags,
                                      nigiri::route_idx_t const route,
                                      std::uint64_t const route_idx,
                                      nigiri::clasz const clasz) {
  auto const lang = nigiri::lang_t{};
  auto const stop_count =
      static_cast<nigiri::stop_idx_t>(tt.route_location_seq_[route].size());

  auto route_infos =
      std::set<std::tuple<std::string, std::string, std::string>>{};
  auto trip_ids = boost::json::array{};

  for (auto const transport_idx : tt.route_transport_ranges_[route]) {
    auto const fr = nigiri::rt::frun{
        tt, nullptr,
        nigiri::rt::run{
            .t_ = nigiri::transport{transport_idx, nigiri::day_idx_t{0}},
            .stop_range_ = nigiri::interval{nigiri::stop_idx_t{0U}, stop_count},
            .rt_ = nigiri::rt_transport_idx_t::invalid()}};

    auto const first = fr[nigiri::stop_idx_t{0U}];
    route_infos.emplace(
        std::string{first.get_route_id(nigiri::event_type::kDep)},
        std::string{first.route_short_name(nigiri::event_type::kDep, lang)},
        std::string{first.route_long_name(nigiri::event_type::kDep, lang)});

    trip_ids.emplace_back(
        boost::json::string{tags.id(tt, first, nigiri::event_type::kDep)});
  }

  return boost::json::object{
      {"route_index", route_idx},
      {"route_clasz", to_str(clasz)},
      {"timetable_routes", utl::transform_to<boost::json::array>(
                               route_infos,
                               [](auto const& info) {
                                 auto const& [id, short_name, long_name] = info;
                                 return boost::json::object{
                                     {"id", id},
                                     {"short_name", short_name},
                                     {"long_name", long_name}};
                               })},
      {"trip_ids", std::move(trip_ids)}};
}

}  // namespace

net::reply shapes_debug::operator()(net::route_request const& req, bool) const {
  utl::verify(c_.shapes_debug_api_enabled(),
              "route shapes debug API is disabled");
  utl::verify(
      w_ != nullptr && l_ != nullptr && tt_ != nullptr && tags_ != nullptr,
      "data not loaded");

  auto const url = boost::url_view{req.target()};
  auto const route_idx = parse_route_idx(url.path());
  utl::verify(route_idx < tt_->n_routes(), "invalid route index {} (max={})",
              route_idx, tt_->n_routes() == 0U ? 0U : tt_->n_routes() - 1U);

  auto const route =
      nigiri::route_idx_t{static_cast<nigiri::route_idx_t::value_t>(route_idx)};
  auto const clasz = tt_->route_clasz_[route];

  auto debug_json = route_shape_debug(*w_, *l_, *tt_, route);
  debug_json["caller"] =
      build_caller_data(*tt_, *tags_, route, route_idx, clasz);
  auto payload = osr::gzip_json(debug_json);
  auto const filename =
      fmt::format("r_{}_{}.json.gz", route_idx, to_str(clasz));

  auto res = net::web_server::string_res_t{boost::beast::http::status::ok,
                                           req.version()};
  res.insert(boost::beast::http::field::content_type, "application/gzip");
  res.insert(boost::beast::http::field::content_disposition,
             fmt::format("attachment; filename=\"{}\"", filename));
  res.insert(boost::beast::http::field::access_control_expose_headers,
             "content-disposition");
  res.body() = std::move(payload);
  res.keep_alive(req.keep_alive());
  return res;
}

}  // namespace motis::ep
