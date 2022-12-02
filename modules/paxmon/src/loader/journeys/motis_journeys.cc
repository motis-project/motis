#include "motis/paxmon/loader/journeys/motis_journeys.h"

#include "utl/enumerate.h"
#include "utl/for_each_line_in_file.h"

#include "motis/core/common/logging.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"

#include "motis/paxmon/access/groups.h"
#include "motis/paxmon/loader/journeys/to_compact_journey.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;

namespace motis::paxmon::loader::journeys {

void load_journey(schedule const& sched, universe& uv,
                  capacity_maps const& caps, journey const& j,
                  data_source const& source, std::uint16_t passengers,
                  route_source_flags source_flags) {
  auto const planned_arrival_time = unix_to_motistime(
      sched.schedule_begin_, j.stops_.back().arrival_.schedule_timestamp_);
  auto tpg =
      temp_passenger_group{0,
                           source,
                           passengers,
                           {{0, 1.0F, to_compact_journey(j, sched),
                             planned_arrival_time, 0, source_flags, true}}};
  add_passenger_group(uv, sched, caps, tpg, false);
}

loader_result load_journeys(schedule const& sched, universe& uv,
                            capacity_maps const& caps,
                            std::string const& journey_file) {
  auto result = loader_result{};

  auto add_journey = [&](journey const& j, std::uint64_t primary_ref = 0,
                         std::uint64_t secondary_ref = 0) {
    load_journey(sched, uv, caps, j, data_source{primary_ref, secondary_ref},
                 1);
    ++result.loaded_journeys_;
  };

  auto line_nr = 0ULL;
  utl::for_each_line_in_file(journey_file, [&](std::string const& line) {
    ++line_nr;
    try {
      auto const res_msg = make_msg(line);
      auto const id = static_cast<std::uint32_t>(res_msg->id());
      switch (res_msg->get()->content_type()) {
        case MsgContent_RoutingResponse: {
          auto const res = motis_content(RoutingResponse, res_msg);
          auto const journeys = message_to_journeys(res);
          for (auto const& [sub_id, j] : utl::enumerate(journeys)) {
            add_journey(j, id, static_cast<std::uint32_t>(sub_id));
          }
          break;
        }
        case MsgContent_Connection: {
          auto const res = motis_content(Connection, res_msg);
          auto const j = convert(res);
          add_journey(j, id);
          break;
        }
        default: break;
      }
    } catch (std::system_error const& e) {
      LOG(motis::logging::error)
          << "invalid message: " << e.what() << ": line " << line_nr;
    }
  });

  return result;
}

}  // namespace motis::paxmon::loader::journeys
