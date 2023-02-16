#include "motis/loader/gtfs/trip.h"

#include <algorithm>
#include <numeric>
#include <stack>
#include <tuple>

#include "utl/enumerate.h"
#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/pairwise.h"
#include "utl/parser/csv.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/schedule/time.h"
#include "motis/loader/util.h"

using namespace utl;
using std::get;

namespace motis::loader::gtfs {

std::vector<std::pair<std::vector<trip*>, bitfield>> block::rule_services() {
  utl::verify(!trips_.empty(), "empty block not allowed");

  utl::erase_if(trips_, [](trip const* t) {
    auto const is_empty = t->stop_times_.empty();
    if (is_empty) {
      LOG(logging::warn) << "trip " << t->id_ << " has no stop times";
    }
    return is_empty;
  });

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
           std::size_t line)
    : route_(route),
      service_(service),
      block_{blk},
      id_{std::move(id)},
      headsign_(std::move(headsign)),
      short_name_(std::move(short_name)),
      line_(line) {}

void trip::interpolate() {
  struct bound {
    explicit bound(int t) : min_{t}, max_{t} {}
    int interpolate(unsigned const idx) const {
      auto const p =
          static_cast<double>(idx - min_idx_) / (max_idx_ - min_idx_);
      return static_cast<int>(min_ + std::round((max_ - min_) * p));
    }
    int min_, max_;
    int min_idx_{-1};
    int max_idx_{-1};
  };
  auto bounds = std::vector<bound>{};
  bounds.reserve(stop_times_.size());
  for (auto const& [i, x] : utl::enumerate(stop_times_)) {
    bounds.emplace_back(x.second.arr_.time_);
    bounds.emplace_back(x.second.dep_.time_);
  }

  auto max = 0;
  auto max_idx = 0U;
  utl::verify(max != kInterpolate, "last arrival cannot be interpolated");
  for (auto it = bounds.rbegin(); it != bounds.rend(); ++it) {
    if (it->max_ == kInterpolate) {
      it->max_ = max;
      it->max_idx_ = max_idx;
    } else {
      max = it->max_;
      max_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }

  auto min = 0;
  auto min_idx = 0U;
  utl::verify(min != kInterpolate, "first arrival cannot be interpolated");
  for (auto it = bounds.begin(); it != bounds.end(); ++it) {
    if (it->min_ == kInterpolate) {
      it->min_ = min;
      it->min_idx_ = min_idx;
    } else {
      min = it->max_;
      min_idx = static_cast<unsigned>(&(*it) - &bounds.front()) / 2U;
    }
  }

  for (auto const& [idx, entry] : utl::enumerate(stop_times_)) {
    auto& [_, stop_time] = entry;
    auto const& arr = bounds[2 * idx];
    auto const& dep = bounds[2 * idx + 1];

    if (stop_time.arr_.time_ == kInterpolate) {
      stop_time.arr_.time_ = arr.interpolate(idx);
    }
    if (stop_time.dep_.time_ == kInterpolate) {
      stop_time.dep_.time_ = dep.interpolate(idx);
    }
  }
}

trip::stop_seq trip::stops() const {
  return utl::to_vec(
      stop_times_, [](flat_map<stop_time>::entry_t const& e) -> stop_identity {
        return {e.second.stop_, e.second.arr_.in_out_allowed_,
                e.second.dep_.in_out_allowed_};
      });
}

trip::stop_seq_numbers trip::seq_numbers() const {
  return utl::to_vec(stop_times_,
                     [](flat_map<stop_time>::entry_t const& e) -> unsigned {
                       return e.first;
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

void trip::print_stop_times(std::ostream& out, unsigned const indent) const {
  for (auto const& t : stop_times_) {
    for (auto i = 0U; i != indent; ++i) {
      out << "  ";
    }
    out << std::setw(60) << t.second.stop_->name_ << " [" << std::setw(5)
        << t.second.stop_->id_ << "]: arr: " << format_time(t.second.arr_.time_)
        << ", dep: " << format_time(t.second.dep_.time_) << "\n";
  }
}

void trip::expand_frequencies(
    std::function<void(trip const&, ScheduleRelationship)> const& consumer)
    const {
  utl::verify(frequency_.has_value(), "bad call to trip::expand_frequencies");

  for (auto const& f : frequency_.value()) {
    for (auto start = f.start_time_; start < f.end_time_; start += f.headway_) {
      trip t{*this};

      auto const delta = t.stop_times_.front().dep_.time_ - start;
      for (auto& stop_time : t.stop_times_) {
        stop_time.second.dep_.time_ -= delta;
        stop_time.second.arr_.time_ -= delta;
      }
      consumer(t, f.schedule_relationship_);
    }
  }
}

std::pair<trip_map, block_map> read_trips(loaded_file file,
                                          route_map const& routes,
                                          traffic_days const& services) {
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

void read_frequencies(loaded_file file, trip_map& trips) {
  if (file.empty()) {
    return;
  }

  enum { trip_id, start_time, end_time, headway_secs, exact_times };
  using gtfs_frequency = std::tuple<cstr, cstr, cstr, cstr, cstr, cstr>;
  static const column_mapping<gtfs_frequency> columns = {
      {"trip_id", "start_time", "end_time", "headway_secs", "exact_times"}};

  auto const entries = read<gtfs_frequency>(file.content(), columns);
  for (auto const& [i, freq] : utl::enumerate(entries)) {
    auto const t = std::get<trip_id>(freq).trim();
    auto const trip_it = trips.find(t.to_str());
    if (trip_it == end(trips)) {
      LOG(logging::error) << "frequencies.txt:" << (i + 1)
                          << ": skipping frequency for non-existing trip \""
                          << t.view() << "\"";
      continue;
    }

    auto const headway_secs_str = std::get<headway_secs>(freq);
    auto const headway_secs = parse<int>(headway_secs_str, -1);
    if (headway_secs == -1) {
      LOG(logging::error) << "frequencies.txt:" << (i + 1)
                          << ": skipping frequency with invalid headway_sec=\""
                          << headway_secs_str.view() << "\"";
      continue;
    }

    auto const exact = std::get<exact_times>(freq);
    auto const schedule_relationship = exact.view() == "1"
                                           ? ScheduleRelationship_SCHEDULED
                                           : ScheduleRelationship_UNSCHEDULED;

    auto& frequencies = trip_it->second->frequency_;
    if (!frequencies.has_value()) {
      frequencies = std::vector<frequency>{};
    }
    frequencies->emplace_back(frequency{hhmm_to_min(std::get<start_time>(freq)),
                                        hhmm_to_min(std::get<end_time>(freq)),
                                        (headway_secs / 60),
                                        schedule_relationship});
  }
}

}  // namespace motis::loader::gtfs
