#include "motis/endpoints/map/route_search.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "net/bad_request_exception.h"

#include "utl/enumerate.h"
#include "utl/erase_duplicates.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "adr/normalize.h"

#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/parse_location.h"
#include "motis/timetable/clasz_to_mode.h"

namespace n = nigiri;

namespace motis::ep {

namespace {

constexpr auto const kDefaultNumResults = std::int64_t{10};
constexpr auto const kApiVersion = 5U;

constexpr auto const kExactScore = 300.F;
constexpr auto const kPrefixScore = 200.F;
constexpr auto const kSubstringScore = 100.F;
constexpr auto const kLengthBonus = 50.F;
constexpr auto const kLongNamePenalty = 25.F;
constexpr auto const kDistancePenalty = 20.F;

std::optional<float> match(std::string_view const name,
                           std::string_view const text) {
  if (name.empty()) {
    return std::nullopt;
  }

  auto const base = [&]() -> std::optional<float> {
    if (name == text) {
      return kExactScore;
    } else if (name.starts_with(text)) {
      return kPrefixScore;
    } else if (name.find(text) != std::string_view::npos) {
      return kSubstringScore;
    }
    return std::nullopt;
  }();

  return base.transform([&](float const b) {
    auto const coverage =
        static_cast<float>(text.size()) / static_cast<float>(name.size());
    return b + kLengthBonus * coverage;
  });
}

std::optional<float> name_score(std::string_view const short_name,
                                std::string_view const long_name,
                                std::string_view const text,
                                adr::utf8_normalize_buf_t& buf) {
  auto best = match(adr::normalize(short_name, buf), text);
  auto const long_score =
      match(adr::normalize(long_name, buf), text).transform([](float const s) {
        return s - kLongNamePenalty;
      });
  if (long_score.has_value() && (!best.has_value() || *long_score > *best)) {
    best = long_score;
  }
  return best;
}

double distance_to(geo::box const& b, geo::latlng const& p) {
  if (b.empty()) {
    return 0.0;
  }
  return geo::distance(
      geo::latlng{std::clamp(p.lat_, b.min_.lat_, b.max_.lat_),
                  std::clamp(p.lng_, b.min_.lng_, b.max_.lng_)},
      p);
}

geo::box route_box(n::timetable const& tt,
                   n::shapes_storage const* shapes,
                   n::route_idx_t const r) {
  if (shapes != nullptr) {
    return shapes->get_bounding_box(r);
  }
  auto b = geo::box{};
  for (auto const s : tt.route_location_seq_[r]) {
    b.extend(tt.locations_.coordinates_.at(n::stop{s}.location_idx()));
  }
  return b;
}

std::optional<n::route_idx_t> first_route(n::timetable const& tt,
                                          n::timetable::route_ids const& rids,
                                          n::route_id_idx_t const r_id) {
  for (auto const trip : rids.route_id_trips_[r_id]) {
    auto const transports = tt.trip_transport_ranges_[trip];
    if (!transports.empty()) {
      return tt.transport_route_[transports.front().first];
    }
  }
  return std::nullopt;
}

std::vector<n::route_idx_t> get_routes(n::timetable const& tt,
                                       n::timetable::route_ids const& rids,
                                       n::route_id_idx_t const r_id) {
  auto routes = std::vector<n::route_idx_t>{};
  for (auto const trip : rids.route_id_trips_[r_id]) {
    for (auto const& [t, _] : tt.trip_transport_ranges_[trip]) {
      routes.emplace_back(tt.transport_route_[t]);
    }
  }
  utl::erase_duplicates(routes);
  return routes;
}

struct candidate {
  n::source_idx_t src_;
  n::route_id_idx_t r_id_;
  n::route_idx_t rep_;
  float score_;
  std::vector<n::route_idx_t> routes_;
};

bool better(candidate const& a, candidate const& b) {
  if (a.score_ != b.score_) {
    return a.score_ > b.score_;
  }
  return a.src_ != b.src_ ? a.src_ < b.src_ : a.r_id_ < b.r_id_;
}

void truncate(std::vector<candidate>& candidates, std::size_t const n) {
  if (candidates.size() <= n) {
    std::ranges::sort(candidates, better);
    return;
  }
  std::ranges::partial_sort(
      candidates, begin(candidates) + static_cast<std::ptrdiff_t>(n), better);
  candidates.resize(n);
}

std::vector<candidate> collect_candidates(n::timetable const& tt,
                                          std::string_view const text,
                                          n::lang_t const& lang,
                                          adr::utf8_normalize_buf_t& buf) {
  auto candidates = std::vector<candidate>{};
  for (auto const [s, rids] : utl::enumerate(tt.route_ids_)) {
    auto const src = n::source_idx_t{s};
    for (auto const [i, short_name] :
         utl::enumerate(rids.route_id_short_names_)) {
      auto const r_id = n::route_id_idx_t{i};
      auto const rep = first_route(tt, rids, r_id);
      if (!rep.has_value()) {
        continue;
      }
      auto const score =
          name_score(tt.translate(lang, short_name),
                     tt.translate(lang, rids.route_id_long_names_[r_id]), text,
                     buf);
      if (score.has_value()) {
        candidates.emplace_back(candidate{src, r_id, *rep, *score, {}});
      }
    }
  }
  return candidates;
}

api::routeSearch_response to_response(n::timetable const& tt,
                                      std::vector<candidate> const& candidates,
                                      n::lang_t const& lang) {
  return api::routeSearch_response{
      .routes_ = utl::to_vec(candidates, [&](candidate const& c) {
        auto const& rids = tt.route_ids_[c.src_];
        auto const color = rids.route_id_colors_[c.r_id_];
        auto const provider = rids.route_id_provider_[c.r_id_];
        return api::RouteMatch{
            .mode_ = to_mode(tt.route_clasz_[c.rep_], kApiVersion),
            .transitRoute_ =
                api::TransitRouteInfo{
                    .id_ = std::string{rids.ids_.get(c.r_id_)},
                    .shortName_ = std::string{tt.translate(
                        lang, rids.route_id_short_names_[c.r_id_])},
                    .longName_ = std::string{tt.translate(
                        lang, rids.route_id_long_names_[c.r_id_])},
                    .color_ = n::to_str(color.color_),
                    .textColor_ = n::to_str(color.text_color_)},
            .agencyName_ =
                provider == n::provider_idx_t::invalid()
                    ? std::nullopt
                    : std::optional{std::string{
                          tt.translate(lang, tt.providers_[provider].name_)}},
            .routeIndexes_ =
                utl::to_vec(c.routes_,
                            [](n::route_idx_t const r) {
                              return static_cast<std::int64_t>(to_idx(r));
                            }),
            .score_ = static_cast<double>(c.score_)};
      })};
}

}  // namespace

api::routeSearch_response route_search::operator()(
    boost::urls::url_view const& url) const {
  auto const params = api::routeSearch_params{url.params()};

  auto buf = adr::utf8_normalize_buf_t{};
  auto const text = std::string{adr::normalize(params.text_, buf)};
  utl::verify<net::bad_request_exception>(!text.empty(),
                                          "text must not be empty");

  auto const place = params.place_.and_then([](std::string const& s) {
    auto const parsed = parse_location(s);
    utl::verify<net::bad_request_exception>(parsed.has_value(),
                                            "could not parse place {}", s);
    return std::optional{parsed.value().pos_};
  });

  auto const config_limit =
      static_cast<std::int64_t>(config_.get_limits().route_search_max_results_);
  auto const num_results =
      std::min(params.numResults_.value_or(kDefaultNumResults), config_limit);
  utl::verify<net::bad_request_exception>(num_results >= 1,
                                          "numResults must be >= 1");

  auto candidates = collect_candidates(tt_, text, params.language_, buf);

  auto const place_bias = static_cast<float>(params.placeBias_);
  if (place.has_value() && place_bias != 0.F) {
    for (auto& c : candidates) {
      auto const dist_km =
          distance_to(route_box(tt_, shapes_, c.rep_), *place) / 1000.0;
      c.score_ -= place_bias * kDistancePenalty *
                  static_cast<float>(std::log1p(dist_km));
    }
  }

  truncate(candidates, static_cast<std::size_t>(num_results));

  for (auto& c : candidates) {
    c.routes_ = get_routes(tt_, tt_.route_ids_[c.src_], c.r_id_);
  }

  return to_response(tt_, candidates, params.language_);
}

}  // namespace motis::ep
