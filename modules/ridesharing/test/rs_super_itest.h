#include <cstdint>
#include <string>

#include "gtest/gtest.h"

#include "boost/geometry.hpp"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "motis/module/message.h"
#include "motis/bootstrap/motis_instance.h"

#include "geo/latlng.h"

namespace motis::ridesharing {

void initialize_mocked(bootstrap::motis_instance& instance,
                       double const base_cost);

motis::module::msg_ptr ridesharing_create(int driver, int64_t time_lift_start,
                                          geo::latlng const& start,
                                          geo::latlng const& dest);
motis::module::msg_ptr ridesharing_create(int driver, int64_t time_lift_start,
                                          double destination_lng = 7.7);
motis::module::msg_ptr ridesharing_edges(double lat = 50.8);

motis::module::msg_ptr ridesharing_edges(int64_t t, geo::latlng const& s,
                                         geo::latlng const& d);

motis::module::msg_ptr ridesharing_stats();

motis::module::msg_ptr ridesharing_book(int driver, int time_lift_start,
                                        int passenger = 345);

motis::module::msg_ptr ridesharing_book(int driver, int time_lift_start,
                                        int passenger, geo::latlng const& piu,
                                        geo::latlng const& dro,
                                        uint16_t from_leg, uint16_t to_leg);

motis::module::msg_ptr ridesharing_get_lifts(int id);

}  // namespace motis::ridesharing
