#include "motis/paxmon/loader/journeys/motis_journeys.h"

#include "utl/enumerate.h"
#include "utl/for_each_line_in_file.h"

#include "motis/core/common/logging.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"

#include "motis/paxmon/loader/journeys/to_compact_journey.h"

using namespace motis::module;
using namespace motis::routing;
using namespace motis::logging;

namespace motis::paxmon::loader::journeys {

std::size_t load_journeys(schedule const& sched, paxmon_data& data,
                          std::string const& journey_file) {
  std::size_t journey_count = 0;

  auto add_journey = [&](journey const& j, std::uint64_t primary_ref = 0,
                         std::uint64_t secondary_ref = 0) {
    auto const id =
        static_cast<std::uint64_t>(data.graph_.passenger_groups_.size());
    data.graph_.passenger_groups_.emplace_back(
        std::make_unique<passenger_group>(
            passenger_group{to_compact_journey(j, sched), 1, id,
                            data_source{primary_ref, secondary_ref}}));
    ++journey_count;
  };

  auto line_nr = 0ULL;
  utl::for_each_line_in_file(journey_file, [&](std::string const& line) {
    ++line_nr;
    try {
      auto const res_msg = make_msg(line);
      auto const id = static_cast<std::uint64_t>(res_msg->id());
      switch (res_msg->get()->content_type()) {
        case MsgContent_RoutingResponse: {
          auto const res = motis_content(RoutingResponse, res_msg);
          auto const journeys = message_to_journeys(res);
          for (auto const& [sub_id, j] : utl::enumerate(journeys)) {
            add_journey(j, id, sub_id);
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

  return journey_count;
}

}  // namespace motis::paxmon::loader::journeys
