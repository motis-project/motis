#include "motis/loader/gtfs/services.h"

#include "boost/date_time/gregorian/gregorian.hpp"

#include "motis/core/common/logging.h"

namespace greg = boost::gregorian;

namespace motis::loader::gtfs {

greg::date bound_date(std::map<std::string, calendar> const& base, bool first) {
  if (base.empty()) {
    return {1970, 1, 1};
  }

  if (first) {
    return std::min_element(begin(base), end(base),
                            [](std::pair<std::string, calendar> const& lhs,
                               std::pair<std::string, calendar> const& rhs) {
                              return lhs.second.first_day_ <
                                     rhs.second.first_day_;
                            })
        ->second.first_day_;
  } else {
    return std::max_element(begin(base), end(base),
                            [](std::pair<std::string, calendar> const& lhs,
                               std::pair<std::string, calendar> const& rhs) {
                              return lhs.second.last_day_ <
                                     rhs.second.last_day_;
                            })
        ->second.last_day_;
  }
}

bitfield calendar_to_bitfield(greg::date const& start, calendar const& c) {
  auto first = std::min(start, c.first_day_);
  auto last =
      std::min(start + greg::days(BIT_COUNT), c.last_day_ + greg::days(1));

  bitfield traffic_days;
  int bit = (first - start).days();
  for (auto d = first; d != last; d += greg::days(1), ++bit) {
    traffic_days.set(bit, c.week_days_.test(d.day_of_week()));
  }
  return traffic_days;
}

void add_exception(greg::date const& start, calendar_date const& exception,
                   bitfield& b) {
  auto day_idx = (exception.day_ - start).days();
  if (day_idx < 0 || day_idx >= static_cast<int>(b.size())) {
    return;
  }
  b.set(day_idx, exception.type_ == calendar_date::ADD);
}

traffic_days merge_traffic_days(
    std::map<std::string, calendar> const& base,
    std::map<std::string, std::vector<calendar_date>> const& exceptions) {
  motis::logging::scoped_timer timer{"traffic days"};

  traffic_days s;
  s.first_day_ = bound_date(base, true);
  s.last_day_ = bound_date(base, false);

  for (auto const& base_calendar : base) {
    s.traffic_days_[base_calendar.first] = std::make_unique<bitfield>(
        calendar_to_bitfield(s.first_day_, base_calendar.second));
  }

  for (auto const& exception : exceptions) {
    for (auto const& day : exception.second) {
      auto bits = s.traffic_days_.find(exception.first);
      if (bits == end(s.traffic_days_)) {
        std::tie(bits, std::ignore) = s.traffic_days_.emplace(
            exception.first, std::make_unique<bitfield>());
      }
      add_exception(s.first_day_, day, *bits->second);
    }
  }

  return s;
}

}  // namespace motis::loader::gtfs
