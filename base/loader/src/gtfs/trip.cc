#include "motis/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parser/csv.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

std::vector<std::pair<std::vector<trip*>, bitfield>> block::rule_services() {
  utl::verify(!trips_.empty(), "empty block not allowed");
  utl::verify(
      std::none_of(begin(trips_), end(trips_),
                   [](trip const* t) { return t->stop_times_.empty(); }),
      "invalid trip with no stop times");
  std::sort(begin(trips_), end(trips_), [](trip const* a, trip const* b) {
    return a->stop_times_.front().dep_.time_ <
           b->stop_times_.front().dep_.time_;
  });

  struct rule_trip {
    trip* trip_;
    bitfield traffic_days_;
  };
  auto rule_trips = utl::to_vec(trips_, [](auto&& t) {
    return rule_trip{t, *t->service_};
  });

  struct queue_entry {
    std::vector<rule_trip>::iterator current_it_;
    std::vector<std::vector<rule_trip>::iterator> collected_trips_;
    bitfield traffic_days_;
  };

  std::vector<std::pair<std::vector<trip*>, bitfield>> combinations;
  for (auto start_it = begin(rule_trips); start_it != end(rule_trips);
       ++start_it) {
    std::stack<queue_entry> q;
    q.emplace(queue_entry{start_it, {}, start_it->traffic_days_});
    while (!q.empty()) {
      auto next = q.top();
      q.pop();

      auto& [current_it, collected_trips, traffic_days] = next;
      collected_trips.emplace_back(current_it);
      for (auto succ_it = std::next(current_it); succ_it != end(rule_trips);
           ++succ_it) {
        if (current_it->trip_->stop_times_.back().stop_ !=
            succ_it->trip_->stop_times_.front().stop_) {
          continue;  // prev last stop != next first stop
        }

        auto const new_intersection = traffic_days & succ_it->traffic_days_;
        traffic_days &= ~succ_it->traffic_days_;
        if (new_intersection.any()) {
          q.emplace(queue_entry{succ_it, collected_trips, new_intersection});
        }
      }

      if (traffic_days.any()) {
        for (auto& rt : collected_trips) {
          rt->traffic_days_ &= ~traffic_days;
        }
        combinations.emplace_back(
            utl::to_vec(collected_trips, [](auto&& rt) { return rt->trip_; }),
            traffic_days);
      }
    }
  }

  return combinations;
}

stop_time::stop_time() = default;

stop_time::stop_time(stop* s, std::string headsign, int arr_time,
                     bool out_allowed, int dep_time, bool in_allowed)
    : stop_{s},
      headsign_{std::move(headsign)},
      arr_{arr_time, out_allowed},
      dep_{dep_time, in_allowed} {}

trip::trip(route const* route, bitfield const* service, block* blk,
           std::string id, std::string headsign, std::string short_name,
           unsigned line)
    : route_(route),
      service_(service),
      block_{blk},
      id_{std::move(id)},
      headsign_(std::move(headsign)),
      short_name_(std::move(short_name)),
      line_(line) {}

trip::stop_seq trip::stops() const {
  return utl::to_vec(
      begin(stop_times_), end(stop_times_),
      [](flat_map<stop_time>::entry_t const& e) -> stop_identity {
        return {e.second.stop_, e.second.arr_.in_out_allowed_,
                e.second.dep_.in_out_allowed_};
      });
}

int trip::avg_speed() const {
  int travel_time = 0.;  // minutes
  double travel_distance = 0.;  // meters

  for (auto const& [dep_entry, arr_entry] : utl::pairwise(stop_times_)) {
    auto const& dep = dep_entry.second;
    auto const& arr = arr_entry.second;
    if (dep.stop_->timezone_ != arr.stop_->timezone_) {
      continue;
    }
    if (arr.arr_.time_ < dep.dep_.time_) {
      continue;
    }

    travel_time += arr.arr_.time_ - dep.dep_.time_;
    travel_distance += geo::distance(dep.stop_->coord_, arr.stop_->coord_);
  }

  return travel_time > 0 ? (travel_distance / 1000.) / (travel_time / 60.) : 0;
}

int trip::distance() const {
  geo::box box;
  for (auto const& [_, stop_time] : stop_times_) {
    box.extend(stop_time.stop_->coord_);
  }
  return geo::distance(box.min_, box.max_) / 1000;
}

enum {
  route_id,
  service_id,
  trip_id,
  trip_headsign,
  trip_short_name,
  block_id
};
using gtfs_trip = std::tuple<cstr, cstr, cstr, cstr, cstr, cstr>;
static const column_mapping<gtfs_trip> columns = {
    {"route_id", "service_id", "trip_id", "trip_headsign", "trip_short_name",
     "block_id"}};

std::pair<trip_map, block_map> read_trips(loaded_file file,
                                          route_map const& routes,
                                          traffic_days const& services) {
  motis::logging::scoped_timer timer{"read trips"};

  std::pair<trip_map, block_map> ret;
  auto& [trips, blocks] = ret;
  auto const entries = read<gtfs_trip>(file.content(), columns);

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Trips").out_bounds(5.F, 25.F).in_high(
      entries.size());
  for (auto const& [i, t] : utl::enumerate(entries)) {
    progress_tracker->update(i);
    auto const blk =
        get<block_id>(t).trim().empty()
            ? nullptr
            : utl::get_or_create(blocks, get<block_id>(t).trim().to_str(),
                                 []() { return std::make_unique<block>(); })
                  .get();
    auto const trp =
        trips
            .emplace(get<trip_id>(t).to_str(),
                     std::make_unique<trip>(
                         routes.at(get<route_id>(t).to_str()).get(),
                         services.traffic_days_.at(get<service_id>(t).to_str())
                             .get(),
                         blk, get<trip_id>(t).to_str(),
                         get<trip_headsign>(t).to_str(),
                         get<trip_short_name>(t).to_str(), i + 1))
            .first->second.get();
    if (blk != nullptr) {
      blk->trips_.emplace_back(trp);
    }
  }
  return ret;
}

}  // namespace motis::loader::gtfs
