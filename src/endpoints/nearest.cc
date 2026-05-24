#include "motis/endpoints/nearest.h"

#include "motis/parse_location.h"

#include "osr/routing/with_profile.h"

namespace motis::ep {

inline std::optional<std::vector<osr::location>> parse_locations(
    std::string_view c) {
  auto parse = [](std::string_view s) -> std::optional<osr::location> {
    auto l = parse_location(s, ',');
    if (!l) return std::nullopt;
    return osr::location{{l->pos_.lng(), l->pos_.lat()}, l->lvl_};
  };

  std::vector<osr::location> locs;
  size_t it;
  while ((it = c.find(';')) != std::string_view::npos) {
    auto l = parse(c.substr(0, it));
    if (!l) return std::nullopt;
    locs.emplace_back(*l);
    c.remove_prefix(it + 1);
  }
  auto l = parse(c);
  if (!l) return std::nullopt;
  locs.emplace_back(*l);
  return locs;
}

api::NearestResponse nearest::operator()(
    boost::urls::url_view const& url) const {
  auto segs = url.segments();
  auto it = segs.begin();
  if (segs.size() < 4) {
    return {.code_ = "InvalidUrl"};
  }
  std::advance(it, 2);

  auto const profile = osr::to_profile(*it++);
  auto const coords = parse_locations(*it++);

  if (!coords || (*coords).empty()) {
    return {.code_ = "InvalidQuery"};
  }

  api::nearest_params param{url.params()};

  if (param.number_ < 1) {
    return {.code_ = "InvalidValue"};
  }

  api::NearestResponse res;
  auto const from = (*coords)[0];
  auto const candidates = osr::with_profile(profile, [&]<typename P>(P&&) {
    constexpr auto kDefaultRadius = 100U;
    auto const max_dist = param.radiuses_.value_or(kDefaultRadius);
    return l_.match<P>(
        std::get<typename P::parameters>(osr::get_parameters(profile)), from,
        false, osr::direction::kForward, max_dist, nullptr);
  });

  auto const n =
      std::min(static_cast<std::size_t>(param.number_), candidates.size());
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
