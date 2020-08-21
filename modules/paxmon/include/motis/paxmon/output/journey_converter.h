#pragma once

#include <cstdint>
#include <functional>

#include "motis/core/journey/journey.h"

#include "motis/paxmon/csv_writer.h"

namespace motis::paxmon::output {

struct journey_converter {
  explicit journey_converter(std::string const& output_path);

  void write_journey(journey const& j, std::uint64_t primary_id,
                     std::uint64_t secondary_id = 0, std::uint16_t pax = 1);

  csv_writer writer_;
};

void for_each_leg(journey const& j,
                  std::function<void(journey::stop const&, journey::stop const&,
                                     extern_trip const&,
                                     journey::transport const*)> const& trip_cb,
                  std::function<void(journey::stop const&,
                                     journey::stop const&)> const& foot_cb);

}  // namespace motis::paxmon::output
