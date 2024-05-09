#include "motis/core/journey/check_journey.h"

namespace motis {

bool check_journey(journey const& j,
                   std::function<std::ostream&(bool)> const& report_error) {
  auto broken = false;
  auto const raise_error = [&]() -> std::ostream& {
    auto const first_error = !broken;
    broken = true;
    return report_error(first_error);
  };

  if (j.stops_.size() >= 2) {
    std::vector<int> segment_transports(j.stops_.size() - 1);
    for (auto i = 0UL; i < j.transports_.size(); ++i) {
      auto const& t = j.transports_[i];
      if (t.to_ <= t.from_) {
        raise_error() << "  Transport " << i << " broken: " << t.from_ << " -> "
                      << t.to_ << '\n';
      }
      for (auto s = t.from_; s < t.to_; ++s) {
        ++segment_transports[s];
      }
    }
    for (auto i = 0UL; i < j.stops_.size() - 1; ++i) {
      if (segment_transports[i] < 1) {
        raise_error() << "  No transport for segment between stops " << i
                      << " -> " << (i + 1) << '\n';
      }
    }
  } else if (j.stops_.size() == 1) {
    raise_error() << "  Connection only has one stop" << '\n';
  }

  for (auto i = 1UL; i < j.stops_.size(); ++i) {
    if (j.stops_[i].arrival_.timestamp_ <
        j.stops_[i - 1].departure_.timestamp_) {
      raise_error() << "  Stops broken: " << (i - 1) << "/" << i
                    << ": Arrival before departure" << '\n';
    }
  }

  for (auto const& stop : j.stops_) {
    if (stop.arrival_.valid_ && stop.departure_.valid_) {
      if (stop.departure_.timestamp_ < stop.arrival_.timestamp_) {
        raise_error() << "  Stop broken: " << stop.name_
                      << ": Departure before arrival" << '\n';
      }
    }
  }

  return !broken;
}

}  // namespace motis
