#include "motis/loader/gtfs/services.h"

#include "boost/date_time/gregorian/gregorian.hpp"

#include "motis/core/common/logging.h"

namespace greg = boost::gregorian;

namespace motis::loader::gtfs {

greg::date bound_date(
    std::map<std::string, calendar> const& base,
    std::map<std::string, std::vector<calendar_date>> const& exceptions,
    bool first) {
  constexpr auto const kMin = greg::date{9999, 12, 31};
  constexpr auto const kMax = greg::date{1400, 1, 1};

  auto const min_base_day = [&]() {
    auto const it =
        std::min_element(begin(base), end(base),
                         [](std::pair<std::string, calendar> const& lhs,
                            std::pair<std::string, calendar> const& rhs) {
                           return lhs.second.first_day_ < rhs.second.first_day_;
                         });
    return it == end(base) ? kMin : it->second.first_day_;
  };

  auto const max_base_day = [&]() {
    auto const it =
        std::max_element(begin(base), end(base),
                         [](std::pair<std::string, calendar> const& lhs,
                            std::pair<std::string, calendar> const& rhs) {
                           return lhs.second.last_day_ < rhs.second.last_day_;
                         });
    return it == end(base) ? kMax : it->second.last_day_;
  };

  if (first) {
    auto min = min_base_day();
    for (auto const& [id, dates] : exceptions) {
      for (auto const& date : dates) {
        if (date.type_ == calendar_date::ADD) {
          min = std::min(min, date.day_);
        }
      }
    }
    return min;
  } else {
    auto max = max_base_day();
    for (auto const& [id, dates] : exceptions) {
      for (auto const& date : dates) {
        if (date.type_ == calendar_date::ADD) {
          max = std::max(max, date.day_);
        }
      }
    }
    return max;
  }
}

bitfield calendar_to_bitfield(std::string const& service_name,
                              greg::date const& start, calendar const& c) {
  auto first = std::min(start, c.first_day_);
  auto last =
      std::min(start + greg::days(BIT_COUNT), c.last_day_ + greg::days(1));

  bitfield traffic_days;
  auto bit = (first - start).days();
  for (auto d = first; d != last; d += greg::days(1), ++bit) {
    if (bit >= traffic_days.size()) {
      LOG(logging::error) << "date " << d << " for service " << service_name
                          << " out of range\n";
      continue;
    }
    traffic_days.set(bit, c.week_days_.test(d.day_of_week()));
  }
  return traffic_days;
}

void add_exception(std::string const& service_name, greg::date const& start,
                   calendar_date const& exception, bitfield& b) {
  auto const day_idx = (exception.day_ - start).days();
  if (day_idx < 0 || day_idx >= static_cast<int>(b.size())) {
    LOG(logging::error) << "date " << exception.day_ << " for service "
                        << service_name << " out of range\n";
    return;
  }
  b.set(day_idx, exception.type_ == calendar_date::ADD);
}

traffic_days merge_traffic_days(
    std::map<std::string, calendar> const& base,
    std::map<std::string, std::vector<calendar_date>> const& exceptions) {
  motis::logging::scoped_timer timer{"traffic days"};

  traffic_days s;
  s.first_day_ = bound_date(base, exceptions, true);
  s.last_day_ = bound_date(base, exceptions, false);

  for (auto const& [service_name, calendar] : base) {
    s.traffic_days_[service_name] = std::make_unique<bitfield>(
        calendar_to_bitfield(service_name, s.first_day_, calendar));
  }

  for (auto const& [service_name, service_exceptions] : exceptions) {
    for (auto const& day : service_exceptions) {
      auto bits = s.traffic_days_.find(service_name);
      if (bits == end(s.traffic_days_)) {
        std::tie(bits, std::ignore) =
            s.traffic_days_.emplace(service_name, std::make_unique<bitfield>());
      }
      add_exception(service_name, s.first_day_, day, *bits->second);
    }
  }

  return s;
}

}  // namespace motis::loader::gtfs
