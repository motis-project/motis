#include "motis/endpoints/one_to_all.h"

#include <chrono>
#include <algorithm>
#include <iostream>  // TODO Remove
#include <variant>
#include <vector>

#include "motis/place.h"
#include "utl/erase_if.h"
#include "utl/verify.h"

#include "adr/types.h"
#include "adr/guess_context.h"

#include "nigiri/common/delta_t.h"
#include "nigiri/for_each_meta.h"
#include "nigiri/location_match_mode.h"
#include "nigiri/routing/one_to_all.h"
#include "nigiri/routing/query.h"
#include "nigiri/types.h"

#include "motis-api/motis-api.h"
#include "motis/gbfs/routing_data.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"

namespace json = boost::json;

namespace motis::ep {
api::Reachable one_to_all::operator()(
    boost::urls::url_view const& url) const {
        auto const query = api::oneToAll_params{url.params()};
    std::cout << "URL: " << url << "\n";
    std::cout << query.one_ << ", " << query.max_ << "\n";

  auto const time = std::chrono::time_point_cast<std::chrono::minutes>(*query.time_.value_or(openapi::now()));
  auto const l = tags_.get_location(tt_, query.one_);
  // utl::verify(l < tt_.n_locations(), "location_idx_t >= n_location: {} >= {}", l, tt_.n_locations());
  auto const pos = tt_.locations_.coordinates_[l];

  auto const start = std::vector<nigiri::routing::offset>{{{l, nigiri::duration_t{}, nigiri::transport_mode_id_t{0U}}}};
  for (auto const s : start) {
    std::cout << "Start: " << s.target() << ", " << s.duration() << "\n";
  }
  auto const q = nigiri::routing::query{
      .start_time_ = time,
      // .start_match_mode_ = nigiri::routing::location_match_mode::kExact,
      .start_match_mode_ = nigiri::routing::location_match_mode::kEquivalent,
      .start_ = std::move(start),
      .max_travel_time_ = nigiri::duration_t{query.max_},
  };
  std::cout << "QUERY BUILT" << std::endl;

  auto const state = [&]() {
    if (query.arriveBy_) {
      return nigiri::routing::one_to_all<nigiri::direction::kBackward>(tt_, nullptr, q);
    } else {
      return nigiri::routing::one_to_all<nigiri::direction::kForward>(tt_, nullptr, q);
    }
  }();

  auto const one = api::Place{
    .name_ = std::string{tt_.locations_.names_[l].view()},
    .stopId_ = tags_.id(tt_, l),
    .lat_ = pos.lat(),
    .lon_ = pos.lng(),
    .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, l))),
    .departure_ = time,
  };
  auto count = 0;
  // auto parents = nigiri::bitvec{tt_.n_locations()};
  // parents.zero_out();
  auto const unreachable = query.arriveBy_ ? nigiri::kInvalidDelta<nigiri::direction::kBackward> : nigiri::kInvalidDelta<nigiri::direction::kForward>;

  auto all = std::vector<api::ReachablePlace>{};
  for (auto i = nigiri::location_idx_t{0U}; i < tt_.n_locations(); ++i) {
    if (state.get_best<0>()[to_idx(i)][0] != unreachable) {
      std::cout << "Reachable: " << i << " (" << tt_.locations_.names_[nigiri::location_idx_t{i}].view() << ")\n";
      auto const dst = tt_.locations_.coordinates_[i];
      auto const fastest = nigiri::routing::get_fastest_one_to_all_offsets<nigiri::direction::kForward>(tt_, state, i, time, q.max_transfers_);
      // auto x = api::ReachablePlace{api::Place{
      //   .name_ = std::string{tt_.locations_.names_[i].view()},
      //   .lat_ = dst.lat(),
      //   .lon_ = dst.lng(),
      //   .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, i))),
      //   .arrival_ = time + std::chrono::minutes{fastest.duration_},
      // },
      // fastest.duration_,
      // fastest.k_,
      // };
      all.emplace_back(api::Place{
        .name_ = std::string{tt_.locations_.names_[i].view()},
        .stopId_ = tags_.id(tt_, i),
        .lat_ = dst.lat(),
        .lon_ = dst.lng(),
        .level_ = static_cast<double>(to_idx(get_lvl(w_, pl_, matches_, i))),
        .arrival_ = time + std::chrono::minutes{fastest.duration_},
      },
      fastest.duration_,
      fastest.k_);
      ++count;
      // std::cout << "Matching for l = " << i << ": ";
      // nigiri::routing::for_each_meta(tt_, nigiri::routing::location_match_mode::kEquivalent, i, [&](nigiri::location_idx_t const l2) {
      //   std::cout << l2 << "(" << static_cast<int>(tt_.locations_.types_[l2]) << "), ";
      // });
      // std::cout << "\n";
      // auto const parent = tt_.locations_.parents_[i] == nigiri::location_idx_t::invalid() ? i : tt_.locations_.parents_[i];
      // parents.set(to_idx(parent), true);
    }
  }
  std::cout << "Counted: " << count << "\n";
  // std::cout << "Reachable parents: " << parents.count() << "\n";
    return {
      .one_ = std::move(one),
      .all_ = std::move(all),
    };
    }

}
