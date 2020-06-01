#include "motis/loader/gtfs/trip.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parser/csv.h"
#include "utl/pipes.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/time.h"
#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

struct rule_trip {
  trip* trip_;
  bitfield traffic_days_;
};

std::vector<std::pair<std::vector<trip*>, bitfield>> block::rule_services() {
  utl::verify(!trips_.empty(), "empty block not allowed");
  std::sort(begin(trips_), end(trips_), [](trip const* a, trip const* b) {
    return a->stop_times_.front().dep_.time_ <
           b->stop_times_.front().dep_.time_;
  });
  auto trips = utl::to_vec(trips_, [](auto&& t) {
    return rule_trip{t, *t->service_};
  });

  struct queue_entry {
    std::vector<rule_trip>::iterator current_it_;
    std::vector<trip*> collected_trips_;
    bitfield intersection_;
  };

  std::queue<queue_entry> q;
  for (auto it = begin(trips); it != end(trips); ++it) {
    q.emplace(queue_entry{it, std::vector<trip*>{}, it->traffic_days_});
  }

  std::vector<std::pair<std::vector<trip*>, bitfield>> combinations;
  while (!q.empty()) {
    auto next = q.front();
    q.pop();

    auto& [current_it, collected_trips, intersection] = next;
    collected_trips.emplace_back(current_it->trip_);
    for (auto succ_it = std::next(current_it); succ_it != end(trips);
         ++succ_it) {
      if (current_it->trip_->stop_times_.back().stop_ !=
          succ_it->trip_->stop_times_.front().stop_) {
        continue;  // prev last stop != next first stop
      }

      auto const new_intersection = intersection & succ_it->traffic_days_;
      if (new_intersection.none()) {
        continue;  // no common traffic days
      }

      current_it->traffic_days_ &= ~new_intersection;
      q.emplace(queue_entry{succ_it, collected_trips, new_intersection});
    }

    if (current_it->traffic_days_.any()) {
      combinations.emplace_back(collected_trips, current_it->traffic_days_);
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
  return utl::to_vec(begin(stop_times_), end(stop_times_),
                     [](flat_map<stop_time>::entry_t const& e) {
                       return std::make_tuple(e.second.stop_,
                                              e.second.arr_.in_out_allowed_,
                                              e.second.dep_.in_out_allowed_);
                     });
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
                                          services const& services) {
  motis::logging::scoped_timer timer{"read trips"};

  std::pair<trip_map, block_map> ret;
  auto& [trips, blocks] = ret;
  auto const entries = read<gtfs_trip>(file.content(), columns);
  motis::logging::clog_import_step("Trips", 5, 20, entries.size());
  for (auto const& [i, t] : utl::enumerate(entries)) {
    motis::logging::clog_import_progress(i, 10000);
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
