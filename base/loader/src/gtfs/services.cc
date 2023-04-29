#include "motis/loader/gtfs/services.h"

#include "motis/core/common/logging.h"

namespace motis::loader::gtfs {

enum class bound { first, last };

date::sys_days bound_date(
    std::map<std::string, calendar> const& base,
    std::map<std::string, std::vector<calendar_date>> const& exceptions,
    bound const b) {
  constexpr auto const kMin = date::sys_days::max();
  constexpr auto const kMax = date::sys_days::min();

  auto const min_base_day = [&]() {
    auto const it =
        std::min_element(begin(base), end(base),
                         [](std::pair<std::string, calendar> const& lhs,
                            std::pair<std::string, calendar> const& rhs) {
                           return lhs.second.first_day_ < rhs.second.first_day_;
                         });
    return it == end(base) ? std::pair{"", kMin}
                           : std::pair{it->first, it->second.first_day_};
  };

  auto const max_base_day = [&]() {
    auto const it =
        std::max_element(begin(base), end(base),
                         [](std::pair<std::string, calendar> const& lhs,
                            std::pair<std::string, calendar> const& rhs) {
                           return lhs.second.last_day_ < rhs.second.last_day_;
                         });
    return it == end(base) ? std::pair{"", kMax}
                           : std::pair{it->first, it->second.last_day_};
  };

  switch (b) {
    case bound::first: {
      auto [min_id, min] = min_base_day();
      for (auto const& [id, dates] : exceptions) {
        for (auto const& date : dates) {
          if (date.type_ == calendar_date::ADD && date.day_ < min) {
            min = date.day_;
            min_id = id;
          }
        }
      }

      LOG(logging::info) << "first date " << date::format("%T", min)
                         << " from service " << min_id;

      return min;
    }
    case bound::last: {
      auto [max_id, max] = max_base_day();
      for (auto const& [id, dates] : exceptions) {
        for (auto const& date : dates) {
          if (date.type_ == calendar_date::ADD && date.day_ > max) {
            max = date.day_;
            max_id = id;
          }
        }
      }

      LOG(logging::info) << "last date " << date::format("%T", max)
                         << " from service " << max_id;

      return max;
    }
  }

  assert(false);
  throw std::runtime_error{"unreachable"};
}

bitfield calendar_to_bitfield(std::string const& service_name,
                              date::sys_days const& start, calendar const& c) {
  bitfield traffic_days;
  auto bit = (c.first_day_ - start).count();
  for (auto d = c.first_day_; d != c.last_day_ + date::days(1);
       d += date::days(1), ++bit) {
    if (bit >= traffic_days.size()) {
      LOG(logging::error) << "date " << date::format("%T", d) << " for service "
                          << service_name << " out of range\n";
      continue;
    }
    auto const weekday_index =
        date::year_month_weekday{d}.weekday().c_encoding();
    traffic_days.set(bit, c.week_days_.test(weekday_index));
  }
  return traffic_days;
}

void add_exception(std::string const& service_name, date::sys_days const& start,
                   calendar_date const& exception, bitfield& b) {
  auto const day_idx = (exception.day_ - start).count();
  if (day_idx < 0 || day_idx >= static_cast<int>(b.size())) {
    LOG(logging::error) << "date " << date::format("%T", exception.day_)
                        << " for service " << service_name << " out of range\n";
    return;
  }
  b.set(day_idx, exception.type_ == calendar_date::ADD);
}

traffic_days merge_traffic_days(
    std::map<std::string, calendar> const& base,
    std::map<std::string, std::vector<calendar_date>> const& exceptions) {
  motis::logging::scoped_timer const timer{"traffic days"};

  traffic_days s;
  s.first_day_ = bound_date(base, exceptions, bound::first);
  s.last_day_ = bound_date(base, exceptions, bound::last);

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
