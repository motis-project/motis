#include "motis/core/journey/print_trip.h"

#include <codecvt>
#include <iomanip>

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/print_journey.h"

namespace motis {

std::ostream& operator<<(std::ostream& out, extern_trip const& ext_trp) {
  return out << "#/trip/" << ext_trp.station_id_ << "/" << ext_trp.train_nr_
             << "/" << ext_trp.time_ << "/" << ext_trp.target_station_id_ << "/"
             << ext_trp.target_time_ << "/" << ext_trp.line_id_;
}

inline std::ostream& print_time(std::ostream& out, std::time_t t,
                                bool local_time) {
  struct tm time {};
#ifdef _MSC_VER
  if (local_time) {
    localtime_s(&time, &t);
  } else {
    gmtime_s(&time, &t);
  }
#else
  if (local_time) {
    localtime_r(&t, &time);
  } else {
    gmtime_r(&t, &time);
  }
#endif
  return out << std::put_time(&time, "%d.%m. %H:%M");
}

void print_trip(std::ostream& out, schedule const& sched, trip const* trp,
                bool const local_time) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> utf8_conv;

  auto i = 0U;
  out << to_extern_trip(sched, trp) << "\n";
  for (auto const& stop : motis::access::stops{trp}) {
    auto const& stop_name = stop.get_station(sched).name_;
    auto const stop_name_len = utf8_conv.from_bytes(stop_name.str()).size();
    std::cout << std::right << std::setw(2) << i << ": " << std::left
              << std::setw(7) << stop.get_station(sched).eva_nr_ << " "
              << std::left
              << std::setw(std::max(0, 50 - static_cast<int>(stop_name_len) +
                                           static_cast<int>(stop_name.size())))
              << std::setfill('.') << stop_name << std::setfill(' ') << " a: ";

    if (!stop.is_first()) {
      print_time(out, motis_to_unixtime(sched, stop.arr_lcon().a_time_),
                 local_time);
    } else {
      out << "            ";
    }
    out << "  d: ";
    if (!stop.is_last()) {
      print_time(out, motis_to_unixtime(sched, stop.dep_lcon().d_time_),
                 local_time);
    }
    out << "\n";

    ++i;
  }
}

}  // namespace motis