#include <chrono>

#include "date/date.h"

#include "gtfsrt/gtfs-realtime.pb.h"

#include "nigiri/types.h"

namespace motis::test {

using namespace std::string_view_literals;
using namespace date;
using namespace std::chrono_literals;

struct trip_descriptor {
  std::string trip_id_;
  std::optional<std::string> start_time_;
  std::optional<std::string> date_;
};

struct trip_update {
  struct stop_time_update {
    std::string stop_id_;
    std::optional<std::uint32_t> seq_{std::nullopt};
    ::nigiri::event_type ev_type_{::nigiri::event_type::kDep};
    std::int32_t delay_minutes_{0U};
    bool skip_{false};
    std::optional<std::string> stop_assignment_{std::nullopt};
  };

  trip_descriptor trip_;
  std::vector<stop_time_update> stop_updates_{};
  bool cancelled_{false};
};

struct alert {
  struct entity_selector {
    std::optional<std::string> agency_id_{};
    std::optional<std::string> route_id_{};
    std::optional<std::int32_t> route_type_{};
    std::optional<std::uint32_t> direction_id_{};
    std::optional<trip_descriptor> trip_{};
    std::optional<std::string> stop_id_{};
  };
  std::string header_;
  std::string description_;
  std::vector<entity_selector> entities_;
};

using feed_entity = std::variant<trip_update, alert>;

template <typename T>
std::uint64_t to_unix(T&& x) {
  return static_cast<std::uint64_t>(
      std::chrono::time_point_cast<std::chrono::seconds>(x)
          .time_since_epoch()
          .count());
};

transit_realtime::FeedMessage to_feed_msg(
    std::vector<feed_entity> const& feed_entities,
    date::sys_seconds const msg_time);

}  // namespace motis::test