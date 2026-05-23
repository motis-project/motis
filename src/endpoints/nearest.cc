#include "motis/endpoints/nearest.h"

#include "osr/routing/map_matching.h"
#include "osr/routing/parameters.h"
#include "osr/routing/with_profile.h"

namespace motis::ep {

inline std::vector<osr::location> parse_array(std::string_view c) {
  auto parse = [&](auto const c) {
    double lat, lon;
    auto const sep = c.find(',');
    std::from_chars(c.data(), c.data() + sep, lon);
    std::from_chars(c.data() + sep + 1, c.data() + c.size(), lat);
    return osr::location{{lat, lon}, osr::kNoLevel};
  };

  std::vector<osr::location> locs;
  size_t it;
  while ((it = c.find(';')) != std::string_view::npos) {
    locs.emplace_back(parse(c.substr(0, it)));
    c.remove_prefix(it + 1);
  }
  locs.emplace_back(parse(c));
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
  auto const coords = parse_array(*it++);

  if (coords.size() != 1) {
    return {.code_ = "InvalidUrl"};
  }

  auto const from = coords[0];

  api::NearestResponse res;
  api::nearest_params param{url.params()};

  auto const candidates = osr::with_profile(profile, [&]<typename P>(P&&) {
    constexpr auto kDefaultRadius = 100U;
    auto const r = param.radiuses_;
    auto const max_dist = r && !r->empty() ? (*r)[0] : kDefaultRadius;
    return l_.match<P>(
        std::get<typename P::parameters>(osr::get_parameters(profile)),
        coords[0], false, osr::direction::kForward, max_dist, nullptr);
  });

  for (auto i = 0U; i < candidates.size(); ++i) {
    auto const c = candidates[i];
    auto const loc = c.closest_point_on_way_;
    auto const id = w_.way_names_[c.way_];
    auto wp = api::Waypoint{};
    if (id != osr::string_idx_t::invalid()) {
      wp.name_ = w_.strings_[id].view();
    }
    wp.distance_ = geo::distance(loc, from.pos_);
    wp.location_ = {loc.lng(), loc.lat()};
  }

  res.code_ = "OK";
  return res;
}

}  // namespace motis::ep
