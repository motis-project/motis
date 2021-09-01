#pragma once

#include <ctime>
#include <functional>
#include <string_view>
#include <vector>

#ifdef NO_DATA
#undef NO_DATA
#endif

#include "motis/core/common/unixtime.h"
#include "motis/ris/ris_message.h"

#include "motis/protocol/RISMessage_generated.h"
#include "gtfsrt.pb.h"

namespace motis {
struct schedule;
}  // namespace motis

namespace motis::ris::gtfsrt {
struct knowledge_context;

struct gtfsrt_parser {
  gtfsrt_parser(schedule const& sched);
  ~gtfsrt_parser();

  gtfsrt_parser(gtfsrt_parser const&) = delete;
  gtfsrt_parser& operator=(gtfsrt_parser const&) = delete;

  gtfsrt_parser(gtfsrt_parser&&) = delete;
  gtfsrt_parser& operator=(gtfsrt_parser&&) = delete;

  void to_ris_message(std::string_view,
                      std::function<void(ris_message&&)> const&);
  std::vector<ris_message> parse(std::string_view);

  bool is_addition_skip_allowed_{true};

private:
  void parse_entity(transit_realtime::FeedEntity const&, unixtime message_time,
                    std::function<void(ris_message&&)> const&);

  void parse_trip_updates(transit_realtime::FeedEntity const&, unixtime,
                          std::function<void(ris_message&&)> const&);

  std::unique_ptr<knowledge_context> knowledge_;
  schedule const& sched_;
};

}  // namespace motis::ris::gtfsrt
