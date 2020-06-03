#include <codecvt>
#include <ctime>
#include <algorithm>
#include <iomanip>

#include "motis/core/common/constants.h"
#include "motis/core/common/date_time_util.h"
#include "motis/core/journey/check_journey.h"
#include "motis/core/journey/print_journey.h"

namespace motis {

namespace {

constexpr auto const LONG_TIME_FORMAT = "%d.%m. %H:%M";
constexpr auto const SHORT_TIME_FORMAT = "%H:%M";

inline std::ostream& print_time(std::ostream& out, std::time_t t,
                                bool local_time,
                                char const* time_format = LONG_TIME_FORMAT) {
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
  return out << std::put_time(&time, time_format);
}

inline int rt_width(realtime_format rt_format) {
  switch (rt_format) {
    case realtime_format::NONE: return 0;
    case realtime_format::OFFSET: return 6;  // " +123R"
    case realtime_format::TIME: return 9;  // " [12:34R]"
    default: return 0;
  }
}

inline void print_rt(std::ostream& out, journey::stop::event_info const& ei,
                     bool local_time, realtime_format rt_format) {
  if (ei.valid_ && (ei.timestamp_reason_ != timestamp_reason::SCHEDULE ||
                    ei.timestamp_ != ei.schedule_timestamp_)) {
    switch (rt_format) {
      case realtime_format::NONE: break;
      case realtime_format::OFFSET: {
        auto const flags = out.flags();
        out.setf(std::ios::showpos);
        out << " " << std::setw(4)
            << ((ei.timestamp_ - ei.schedule_timestamp_) / 60)
            << ei.timestamp_reason_;
        out.flags(flags);
        break;
      }
      case realtime_format::TIME: {
        out << " [";
        print_time(out, ei.timestamp_, local_time, SHORT_TIME_FORMAT);
        out << ei.timestamp_reason_ << "]";
        break;
      }
    }
  } else {
    out << std::setw(rt_width(rt_format)) << "";
  }
}

void print_event(std::ostream& out, journey::stop::event_info const& ei,
                 bool local_time, realtime_format rt_format) {
  if (ei.valid_) {
    print_time(out, ei.schedule_timestamp_, local_time);
  } else {
    out << "            ";
  }
  print_rt(out, ei, local_time, rt_format);
}

inline journey::stop::event_info const& departure_event(journey const& j) {
  return j.stops_.front().departure_;
}

inline journey::stop::event_info const& arrival_event(journey const& j) {
  return j.stops_.back().arrival_;
}

inline bool is_virtual_station(journey::stop const& stop) {
  auto const& name = stop.name_;
  return name == STATION_START || name == STATION_END;
}

}  // namespace

bool print_journey(journey const& j, std::ostream& out, bool local_time,
                   realtime_format rt_format) {
  out << "Journey: duration=" << std::left << std::setw(3) << j.duration_
      << " transfers=" << std::left << std::setw(2) << j.transfers_
      << " accessibility=" << std::left << std::setw(3) << j.accessibility_
      << "              ";
  print_event(out, departure_event(j), local_time, rt_format);
  out << " --> ";
  print_event(out, arrival_event(j), local_time, rt_format);
  if (local_time) {
    out << " (local)" << std::endl;
  } else {
    out << " (UTC)" << std::endl;
  }

  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> utf8_conv;
  out << "\nStops:" << std::endl;
  for (auto i = 0UL; i < j.stops_.size(); ++i) {
    auto const& stop = j.stops_[i];
    auto stop_name = is_virtual_station(stop)
                         ? stop.name_ + " (" + std::to_string(stop.lat_) + ";" +
                               std::to_string(stop.lng_) + ")"
                         : stop.name_;
    auto const stop_name_len = utf8_conv.from_bytes(stop_name).size();
    out << std::right << std::setw(2) << i << ": " << std::left << std::setw(7)
        << stop.eva_no_ << " " << std::left
        << std::setw(std::max(0, 50 - static_cast<int>(stop_name_len) +
                                     static_cast<int>(stop_name.size())))
        << std::setfill('.') << stop_name << std::setfill(' ') << " a: ";
    print_event(out, stop.arrival_, local_time, rt_format);
    out << "  d: ";
    print_event(out, stop.departure_, local_time, rt_format);
    if (stop.exit_) {
      out << " exit";
    } else {
      out << "     ";
    }
    if (stop.enter_) {
      out << " enter";
    }
    out << std::endl;
  }

  out << "\nTransports:" << std::endl;
  for (auto i = 0UL; i < j.transports_.size(); ++i) {
    auto const& trans = j.transports_[i];
    out << std::right << std::setw(2) << i << ": " << std::left << std::setw(2)
        << trans.from_ << " -> " << std::left << std::setw(2) << trans.to_
        << " " << (trans.is_walk_ ? "WALK  " : "TRAIN ") << std::left;
    if (trans.is_walk_) {
      out << "type=" << std::left << std::setw(9) << trans.mumo_type_
          << " id=" << std::left << std::setw(7) << trans.mumo_id_
          << " duration=" << std::left << std::setw(3) << trans.duration_
          << " accessibility=" << std::left << std::setw(3)
          << trans.mumo_accessibility_ << std::endl;
    } else {
      out << std::left << std::setw(10) << trans.name_
          << "                duration=" << std::left << std::setw(3)
          << trans.duration_ << std::endl;
    }
  }

  out << "\nTrips:" << std::endl;
  for (auto i = 0UL; i < j.trips_.size(); ++i) {
    auto const& trp = j.trips_[i].extern_trip_;
    out << std::right << std::setw(2) << i << ": " << std::left << std::setw(2)
        << j.trips_[i].from_ << " -> " << std::left << std::setw(2)
        << j.trips_[i].to_ << " {" << std::setw(7) << trp.station_id_ << ", "
        << std::right << std::setw(6) << trp.train_nr_ << ", "
        << format_unix_time(trp.time_) << "} -> {" << std::setw(7)
        << trp.target_station_id_ << ", " << format_unix_time(trp.target_time_)
        << "}, line_id=" << trp.line_id_ << "\n       #/trip/"
        << trp.station_id_ << "/" << trp.train_nr_ << "/" << trp.time_ << "/"
        << trp.target_station_id_ << "/" << trp.target_time_ << "/"
        << trp.line_id_ << "  " << j.trips_[i].debug_ << std::endl;
  }

  auto const report_error = [&](bool first_error) -> std::ostream& {
    if (first_error) {
      out << "\nWARNING: Journey is broken:" << std::endl;
    }
    return out;
  };

  return check_journey(j, report_error);
}

}  // namespace motis
