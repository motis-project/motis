#include "motis/endpoints/nearest.h"

#include "motis/parse_location.h"

#include "osr/routing/with_profile.h"

#include "net/bad_request_exception.h"

namespace motis::ep {

api::NearestResponse nearest::operator()(
    boost::urls::url_view const& url) const {

  auto parse_params = [&](auto const s,
                          auto const p) -> std::optional<api::nearest_params> {
    api::nearest_params params{p};
    auto it = s.begin();
    if (s.size() < 4) {
      return std::nullopt;
    }
    std::advance(it, 2);
    params.profile_ = (*it++);
    params.coordinates_ = (*it++);
    return params;
  };

  auto const params = parse_params(url.segments(), url.params());
  utl::verify<net::bad_request_exception>(params.has_value(),
                                          "invalid path segments");

  std::optional<osr::search_profile> profile{};
  std::optional<osr::location> coord{};

  try {
    profile = osr::to_profile(*params->profile_);
  } catch (...) {
    throw net::bad_request_exception("invalid profile");
  }

  coord = parse_location(*params->coordinates_);
  utl::verify<net::bad_request_exception>(coord.has_value(),
                                          "invalid coordinates");
  utl::verify<net::bad_request_exception>(params->number_ >= 1,
                                          "invalid number");

  std::swap(coord->pos_.lat_, coord->pos_.lng_);
  api::NearestResponse res;
  auto const from = coord.value();
  auto const candidates = osr::with_profile(*profile, [&]<typename P>(P&&) {
    constexpr auto kDefaultRadius = 100U;
    auto const max_dist = params->radiuses_.value_or(kDefaultRadius);
    return l_.match<P>(
        std::get<typename P::parameters>(osr::get_parameters(*profile)), from,
        false, osr::direction::kForward, max_dist, nullptr);
  });

  if (candidates.empty()) {
    res.code_ = "NoSegment";
    return res;
  }

  auto const n =
      std::min(static_cast<std::size_t>(params->number_), candidates.size());
  for (auto i = 0U; i < n; ++i) {
    auto const c = candidates[i];
    auto const loc = c.closest_point_on_way_;
    auto const id = w_.way_names_[c.way_];
    auto wp = api::Waypoint{};
    if (id != osr::string_idx_t::invalid()) {
      wp.name_ = w_.strings_[id].view();
    }
    wp.distance_ = geo::distance(loc, from.pos_);
    wp.location_ = {loc.lng(), loc.lat()};
    res.waypoints_.emplace_back(wp);
  }

  res.code_ = "OK";
  return res;
}

}  // namespace motis::ep
