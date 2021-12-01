#include "motis/loader/route.h"

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
                          schedule const& sched) {
  utl::verify(std::all_of(begin(lcons_), end(lcons_),
                          [&new_lcons](auto const& i) {
                            return new_lcons.size() == i.size();
                          }),
              "number of sections not matching");
  utl::verify(new_lcons.size() * 2 == new_times.size(),
              "number of times and lcons not matching");

  auto const insert_it = std::lower_bound(
      begin(times_), end(times_), new_times,
      [](mcd::vector<time> const& lhs, mcd::vector<time> const& rhs) {
        return lhs.front() < rhs.front();
      });
  auto const insert_idx = std::distance(begin(times_), insert_it);

  // check full time array!
  // Example: insert [3,9,14,18] into [[4,10,12,16]]
  // check 3 < 4 -> ok
  // check 9 < 10 -> ok
  // check 14 < 12 -> fail, new service overtakes the existing
  for (unsigned i = 0; i < new_times.size(); ++i) {
    auto middle_time = new_times[i].mam();
    bool before_pred = false;
    bool after_succ = false;
    if (!times_.empty()) {
      if (insert_idx != 0) {
        auto before_time = times_[insert_idx - 1][i].mam();
        before_pred = middle_time <= before_time;
      }
      if (static_cast<int>(insert_idx) < static_cast<int>(times_.size())) {
        auto after_time = times_[insert_idx][i].mam();
        after_succ = middle_time >= after_time;
      }
    }
    if (before_pred || after_succ) {
      return false;
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
