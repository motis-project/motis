#include <ctime>
#include <string>

#include "gtest/gtest.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/access/time_access.h"

namespace motis::loader {

class loader_graph_builder_test : public ::testing::Test {
protected:
  loader_graph_builder_test(std::string schedule_name,
                            std::string schedule_begin, int num_days);

  void SetUp() override;

  static edge const* get_route_edge(node const* route_node);

  static std::vector<
      std::tuple<light_connection const*, day_idx_t, node const*, node const*>>
  get_connections(node const* first_route_node, time departure_time);

  std::time_t unix_time(int hhmm, int day_idx = 0,
                        int timezone_offset = DEFAULT_TIMEZONE_OFFSET) {
    return motis::unix_time(*sched_, hhmm, day_idx, timezone_offset);
  }

  void print_trip(concrete_trip) const;

  int trip_count(std::vector<station const*>) const;
  bool check_trip_path(trip_info const*,
                       std::vector<station const*> const&) const;

  schedule_ptr sched_;
  std::string schedule_name_;
  std::string schedule_begin_;
  int num_days_;
};

}  // namespace motis::loader
