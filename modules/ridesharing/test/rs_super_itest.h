#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"

#include "motis/module/message.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/lift.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

#include <cstdint>

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"
#include "gtest/gtest.h"

namespace motis::ridesharing {

struct rs_super_itest : public motis::test::motis_instance_test {
  explicit rs_super_itest(std::vector<std::string> const& modules,
                          std::vector<std::string> const& options);

  motis::module::msg_ptr ridesharing_create(int driver, int64_t time_lift_start,
                                            geo::latlng const& start,
                                            geo::latlng const& dest);
  motis::module::msg_ptr ridesharing_create(int driver, int64_t time_lift_start,
                                            double destination_lng = 7.7);
  motis::module::msg_ptr ridesharing_edges(double lat = 50.8);

  motis::module::msg_ptr ridesharing_edges(int64_t t, geo::latlng const& s,
                                           geo::latlng const& d);

  motis::module::msg_ptr ridesharing_stats() {
    return motis::module::make_no_msg("/ridesharing/stats");
  }

  motis::module::msg_ptr ridesharing_book(int driver, int time_lift_start,
                                          int passenger = 345);

  motis::module::msg_ptr ridesharing_book(int driver, int time_lift_start,
                                          int passenger, geo::latlng const& piu,
                                          geo::latlng const& dro,
                                          uint16_t from_leg, uint16_t to_leg);

  motis::module::msg_ptr ridesharing_get_lifts(int id);
};

}  // namespace motis::ridesharing
