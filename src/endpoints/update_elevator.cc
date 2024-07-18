#include "icc/endpoints/update_elevator.h"

#include "utl/helpers/algorithm.h"
#include "utl/parallel_for.h"
#include "utl/zip.h"

#include "osr/routing/profiles/foot.h"
#include "osr/routing/route.h"

#include "nigiri/footpath.h"

#include "icc/constants.h"
#include "icc/get_loc.h"

namespace json = boost::json;
namespace n = nigiri;

namespace icc::ep {

void update_elevator::update_elevators(elevators const& pred,
                                       elevators const& next) const {
  auto changed_set = hash_set<n::location_idx_t>{};
  for (auto i = osr::node_idx_t{0U}; i != pred.blocked_.size(); ++i) {
    if (pred.blocked_[i] != next.blocked_[i]) {
      loc_rtree_.in_radius(
          w_.get_node_pos(i), kElevatorUpdateRadius,
          [&](n::location_idx_t const l) { changed_set.emplace(l); });
    }
  }

  struct data {
    std::vector<n::footpath> foot_, wheelchair_;
  };
  auto const changed = utl::to_vec(changed_set, [](auto&& x) { return x; });
  utl::parallel_for_run_threadlocal<data>(changed.size(), [&](data& data,
                                                              auto const i) {
    data.foot_.clear();
    data.wheelchair_.clear();

    auto const l = n::location_idx_t{changed[i]};
    for (auto const mode :
         {osr::search_profile::kFoot, osr::search_profile::kWheelchair}) {
      auto neighbors = std::vector<n::location_idx_t>{};
      loc_rtree_.in_radius(
          tt_.locations_.coordinates_[l], kMaxDistance,
          [&](n::location_idx_t const l) { neighbors.emplace_back(l); });
      auto const results = osr::route(
          w_, l_, mode, get_loc(tt_, w_, pl_, matches_, l),
          utl::to_vec(
              neighbors,
              [&](auto&& l) { return get_loc(tt_, w_, pl_, matches_, l); }),
          kMaxDuration, osr::direction::kForward, kMaxMatchingDistance,
          &next.blocked_);
      for (auto const [n, r] : utl::zip(neighbors, results)) {
        if (r.has_value()) {
          auto const duration = n::duration_t{r->cost_ / 60U};
          if (duration < n::footpath::kMaxDuration) {
            (mode == osr::search_profile::kFoot ? data.foot_ : data.wheelchair_)
                .emplace_back(n::footpath{n, duration});
          }
        }
      }
    }

    utl::sort(data.foot_);
    utl::sort(data.wheelchair_);
  });
}

json::value update_elevator::operator()(json::value const& query) const {
  auto const& q = query.as_object();
  auto const id = q.at("id").to_number<std::int64_t>();
  auto const status = status_from_str(q.at("status").as_string());

  auto const e = e_.get();
  auto elevators_copy = e->elevators_;
  auto const it =
      utl::find_if(elevators_copy, [&](auto&& x) { return x.id_ == id; });
  if (it == end(elevators_copy)) {
    return json::value{{"error", "id not found"}};
  }

  it->status_ = status;

  auto const updated = std::make_shared<elevators>(w_, elevator_nodes_,
                                                   std::move(elevators_copy));

  //  update_elevators(w_, l_, pl_, *e_, *updated);

  e_ = updated;

  return json::string{{"success", true}};
}

}  // namespace icc::ep