#include "motis/loader/route.h"

#include "motis/core/access/track_access.h"

namespace motis::loader {

route_t::route_t() = default;

route_t::route_t(mcd::vector<light_connection> const& new_lcons,
                 mcd::vector<time> const& times, schedule const& sched) {
  lcons_.emplace_back(new_lcons);
  times_.emplace_back(times);
  update_traffic_days(new_lcons, sched);
}

bool route_t::add_service(mcd::vector<light_connection> const& new_lcons,
                          mcd::vector<time> const& new_times,
                          schedule const& sched,
                          mcd::vector<station*> const& stations) {
  utl::verify(std::all_of(begin(lcons_), end(lcons_),
                          [&new_lcons](auto const& i) {
                            return new_lcons.size() == i.size();
                          }),
              "number of sections not matching");
  utl::verify(new_lcons.size() * 2 == new_times.size(),
              "number of times and lcons not matching");

  auto const insert_it = std::lower_bound(
      begin(times_), end(times_), new_times,
      [](mcd::vector<time> const& a, mcd::vector<time> const& b) {
        return a.front() < b.front();
      });
  auto const insert_idx = std::distance(begin(times_), insert_it);

  // check full time array!
  // Example: insert [3,9,14,18] into [[4,10,12,16]]
  // check 3 < 4 -> ok
  // check 9 < 10 -> ok
  // check 14 < 12 -> fail, new service overtakes the existing
  if (!times_.empty()) {
    for (auto i = 0U; i < new_times.size(); ++i) {
      auto const middle_time = new_times[i].mam();

      if (insert_idx != 0) {
        auto const pred_time = times_[insert_idx - 1][i].mam();
        if (middle_time <= pred_time) {
          return false;
        }
      }

      if (insert_idx < times_.size()) {
        auto const succ_time = times_[insert_idx][i].mam();
        if (middle_time >= succ_time) {
          return false;
        }
      }
    }

    for (auto section_idx = 0U; section_idx != new_lcons.size();
         ++section_idx) {
      for (auto day_idx = 0U; day_idx != MAX_DAYS; ++day_idx) {
        if (!sched.bitfields_.at(new_lcons.at(section_idx).traffic_days_)
                 .test(day_idx)) {
          continue;
        }

        auto const d_station = stations[section_idx];
        auto const d_track = get_track_string_idx(
            sched, lcons_.front().at(section_idx).full_con_->d_track_, day_idx);
        auto const new_d_track = get_track_string_idx(
            sched, new_lcons.at(section_idx).full_con_->d_track_, day_idx);
        if (d_station->get_platform(d_track) !=
            d_station->get_platform(new_d_track)) {
          return false;
        }

        auto const a_station = stations[section_idx + 1];
        auto const a_track = get_track_string_idx(
            sched, lcons_.front().at(section_idx).full_con_->a_track_, day_idx);
        auto const new_a_track = get_track_string_idx(
            sched, new_lcons.at(section_idx).full_con_->a_track_, day_idx);
        if (a_station->get_platform(a_track) !=
            a_station->get_platform(new_a_track)) {
          return false;
        }
      }
    }
  }

  // new_s is safe to add at idx
  times_.insert(std::next(begin(times_), insert_idx), new_times);
  lcons_.insert(std::next(begin(lcons_), insert_idx), new_lcons);
  verify_sorted();

  update_traffic_days(new_lcons, sched);

  return true;
}

void route_t::verify_sorted() {
  utl::verify(
      std::is_sorted(begin(times_), end(times_),
                     [](mcd::vector<time> const& a,
                        mcd::vector<time> const& b) { return a[0] < b[0]; }),
      "route services not sorted");
}

bool route_t::empty() const { return times_.empty(); }

void route_t::update_traffic_days(
    mcd::vector<light_connection> const& new_lcons, schedule const& sched) {
  for (auto const& lcon : new_lcons) {
    traffic_days_ |= sched.bitfields_[lcon.traffic_days_.bitfield_idx_];
  }
}

mcd::vector<light_connection> const& route_t::operator[](
    std::size_t idx) const {
  assert(idx < lcons_.size());
  return lcons_[idx];
}

}  // namespace motis::loader
