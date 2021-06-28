#include "motis/paxmon/generate_capacities.h"

#include <cstdint>
#include <algorithm>
#include <fstream>
#include <random>
#include <utility>
#include <vector>

#include "motis/core/common/logging.h"

#include "motis/paxmon/capacity.h"
#include "motis/paxmon/trip_section_load_iterator.h"

using namespace motis::logging;

namespace motis::paxmon {

template <typename Generator>
int get_capacity(Generator& rng, std::normal_distribution<>& dist,
                 std::uint16_t load, int min = 25) {
  auto val = min;
  for (auto i = 0; i < 10; ++i) {
    val = std::max(val, static_cast<int>(dist(rng)));
    if (val >= load) {
      return val;
    }
  }
  return static_cast<int>(dist.mean() + 3 * dist.stddev());
}

void generate_capacities(schedule const& sched, paxmon_data const& data,
                         std::string const& filename) {
  auto rng = std::mt19937{std::random_device{}()};

  auto d_ice1 = std::discrete_distribution<>{.48, .28, .21, .01};
  auto d_ice2 = std::vector<std::normal_distribution<>>{
      std::normal_distribution<>{430, 14},  //
      std::normal_distribution<>{720, 10},  //
      std::normal_distribution<>{840, 20},  //
      std::normal_distribution<>{1000, 10}};
  auto d_ic = std::normal_distribution<>{450, 70};
  auto d_coach = std::normal_distribution<>{40, 5};
  auto d_night = std::normal_distribution<>{300, 40};
  auto d_rerb = std::normal_distribution<>{200, 40};
  auto d_s = std::normal_distribution<>{300, 60};
  auto d_u = std::normal_distribution<>{100, 20};
  auto d_str = std::normal_distribution<>{100, 20};
  auto d_bus = std::normal_distribution<>{50, 20};

  std::ofstream out{filename};
  out.exceptions(std::ios_base::failbit | std::ios_base::badbit);

  out << "train_nr,category,from,to,departure,arrival,seats,base_load,"
         "remaining\n";

  auto generated = 0ULL;
  auto over_capacity = 0ULL;
  for (auto const& tp : sched.trip_mem_) {
    auto const trp = tp.get();
    auto has_capacity_data = true;
    auto max_load = std::uint16_t{0};
    auto const sections = sections_with_load{sched, data, trp};
    if (sections.empty()) {
      continue;
    }
    for (auto const& ts : sections) {
      has_capacity_data = has_capacity_data && ts.get_capacity_source() ==
                                                   capacity_source::TRIP_EXACT;
      max_load = std::max(max_load, ts.base_load());
    }
    if (has_capacity_data) {
      continue;
    }
    auto const clasz = sections.front().section_.lcon().full_con_->clasz_;
    auto capacity = 0;
    switch (clasz) {
      case service_class::ICE:
        for (auto idx = d_ice1(rng); idx < d_ice2.size(); ++idx) {
          capacity = get_capacity(rng, d_ice2[idx], max_load, 250);
          if (capacity >= max_load) {
            break;
          }
        }
        break;
      case service_class::IC:
        capacity = get_capacity(rng, d_ic, max_load, 70);
        break;
      case service_class::COACH:
        capacity = get_capacity(rng, d_coach, max_load, 30);
        break;
      case service_class::N:
        capacity = get_capacity(rng, d_night, max_load, 50);
        break;
      case service_class::RE:
      case service_class::RB:
        capacity = get_capacity(rng, d_rerb, max_load, 50);
        break;
      case service_class::S:
        capacity = get_capacity(rng, d_s, max_load, 50);
        break;
      case service_class::U:
        capacity = get_capacity(rng, d_u, max_load, 50);
        break;
      case service_class::STR:
        capacity = get_capacity(rng, d_str, max_load, 40);
        break;
      case service_class::BUS:
        capacity = get_capacity(rng, d_bus, max_load, 25);
        break;
      default: continue;
    }

    auto const family =
        sections.front().section_.lcon().full_con_->con_info_->family_;
    auto const cap_tid = cap_trip_id{
        static_cast<std::uint32_t>(trp->id_.primary_.train_nr_),
        trp->id_.primary_.get_station_id(),
        trp->id_.secondary_.target_station_id_, trp->id_.primary_.get_time(),
        trp->id_.secondary_.target_time_};
    out << cap_tid.train_nr_ << ","
        << sched.categories_.at(family)->name_.view() << ","
        << sched.stations_.at(cap_tid.from_station_idx_)->eva_nr_.view() << ","
        << sched.stations_.at(cap_tid.to_station_idx_)->eva_nr_.view() << ","
        << motis_to_unixtime(sched.schedule_begin_, cap_tid.departure_) << ","
        << motis_to_unixtime(sched.schedule_begin_, cap_tid.arrival_) << ","
        << capacity << "," << max_load << "," << (capacity - max_load) << "\n";
    ++generated;
    if (max_load > capacity) {
      ++over_capacity;
    }
  }

  LOG(info) << "generated capacity information for " << generated << "/"
            << sched.trip_mem_.size() << " trips, " << over_capacity
            << " over capacity";
}

}  // namespace motis::paxmon
