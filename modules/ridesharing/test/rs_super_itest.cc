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

#include "./rs_super_itest.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "geo/constants.h"
#include "geo/detail/register_latlng.h"
#include "geo/latlng.h"
#include "gtest/gtest.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::parking;
using namespace motis::ppr;
using namespace motis::intermodal;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::ridesharing {

rs_super_itest::rs_super_itest(double base_cost)
    : motis::test::motis_instance_test(dataset_opt, {"lookup", "ridesharing"},
                                       {"--ridesharing.database_path=:memory:",
                                        "--ridesharing.use_parking=false"}),
      base_cost_(base_cost) {
  initialize_mocked();
}

void rs_super_itest::initialize_mocked() {
  instance_->register_op("/osrm/many_to_many", [&](msg_ptr const& msg) {
    message_creator mc;
    auto const req = motis_content(OSRMManyToManyRequest, msg);
    auto const coor = utl::to_vec(*req->coordinates(), [](auto const& loc) {
      return geo::latlng{loc->lat(), loc->lng()};
    });

    std::vector<Offset<OSRMManyToManyResponseRow>> routing_matrix;
    for (auto const& from : coor) {
      std::vector<Cost> cost_row;
      cost_row.reserve(coor.size());
      for (auto const& to : coor) {
        cost_row.push_back(test_routing_cost(from, to));
      }
      routing_matrix.emplace_back(CreateOSRMManyToManyResponseRow(
          mc, mc.CreateVectorOfStructs(cost_row)));
    }

    mc.create_and_finish(
        MsgContent_OSRMManyToManyResponse,
        CreateOSRMManyToManyResponse(mc, mc.CreateVector(routing_matrix))
            .Union());
    return make_msg(mc);
  });

  instance_->register_op("/osrm/one_to_many", [&](msg_ptr const& msg) {
    message_creator mc;
    auto const req = motis_content(OSRMOneToManyRequest, msg);
    auto const one = geo::latlng{req->one()->lat(), req->one()->lng()};
    auto const many = utl::to_vec(*req->many(), [](auto const& loc) {
      return geo::latlng{loc->lat(), loc->lng()};
    });
    auto const costs = utl::to_vec(many, [this, &one](auto const& loc) {
      return test_routing_cost(one, loc);
    });
    mc.create_and_finish(
        MsgContent_OSRMOneToManyResponse,
        CreateOSRMOneToManyResponse(mc, mc.CreateVectorOfStructs(costs))
            .Union());
    return make_msg(mc);
  });

  instance_->register_op("/osrm/route", [&](msg_ptr const& msg) {
    message_creator mc;
    auto const req = motis_content(OSRMRouteRequest, msg);
    auto const coordinates =
        utl::to_vec(*req->coordinates(), [](auto const& co) {
          return geo::latlng{co->lat(), co->lng()};
        });
    std::vector<Cost> costs;
    for (auto it = coordinates.begin(); it + 1 != coordinates.end(); it++) {
      costs.push_back(test_routing_cost(*it, *(it + 1)));
    }

    mc.create_and_finish(
        MsgContent_OSRMRouteResponse,
        CreateOSRMRouteResponse(mc, mc.CreateVectorOfStructs(costs)).Union());
    return make_msg(mc);
  });

  instance_->register_op("/ppr/route", [&](msg_ptr const&) {
    message_creator mc;
    Position start{50.8, 6.6};
    Position target{50.8, 6.7};
    std::vector<Offset<RouteStep>> rs{};
    std::vector<Offset<Edge>> e{};
    std::vector<Offset<Route>> r{CreateRoute(
        mc, 2, 3, 3.14, 3.15, 2, 2.69, 2.68, &start, &target,
        mc.CreateVector(rs), mc.CreateVector(e), CreatePolyline(mc), 0, 0)};

    std::vector<Offset<Routes>> routes{CreateRoutes(mc, mc.CreateVector(r))};
    mc.create_and_finish(
        MsgContent_FootRoutingResponse,
        CreateFootRoutingResponse(mc, mc.CreateVector(routes)).Union());
    return make_msg(mc);
  });
}

msg_ptr rs_super_itest::ridesharing_create(int driver, int64_t time_lift_start,
                                           geo::latlng const& start,
                                           geo::latlng const& dest) {
  message_creator mc;
  Position s{start.lat_, start.lng_};
  Position d{dest.lat_, dest.lng_};

  mc.create_and_finish(
      MsgContent_RidesharingCreate,
      CreateRidesharingCreate(mc, driver, time_lift_start, 4, &s, &d).Union(),
      "/ridesharing/create");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_create(int driver, int64_t time_lift_start,
                                           double destination_lng) {
  message_creator mc;
  Position start{50.8, 6.0};
  Position destination{50.8, destination_lng};

  mc.create_and_finish(MsgContent_RidesharingCreate,
                       CreateRidesharingCreate(mc, driver, time_lift_start, 4,
                                               &start, &destination)
                           .Union(),
                       "/ridesharing/create");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_edges(double const lat) {
  message_creator mc;
  Position start{lat, 6.2};
  Position destination{lat, 7.2};
  std::string profile{"default"};

  mc.create_and_finish(
      MsgContent_RidesharingRequest,
      CreateRidesharingRequest(mc, &start, &destination, 0, 1, QUERYMODE_BOTH,
                               motis::ppr::CreateSearchOptions(
                                   mc, mc.CreateString(profile), 1000000))
          .Union(),
      "/ridesharing/edges");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_edges(int64_t t, geo::latlng const& s,
                                          geo::latlng const& d) {
  message_creator mc;
  Position start{s.lat_, s.lng_};
  Position destination{d.lat_, d.lng_};
  std::string profile{"default"};

  mc.create_and_finish(
      MsgContent_RidesharingRequest,
      CreateRidesharingRequest(mc, &start, &destination, t, 1, QUERYMODE_BOTH,
                               motis::ppr::CreateSearchOptions(
                                   mc, mc.CreateString(profile), 1000000))
          .Union(),
      "/ridesharing/edges");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_book(int driver, int time_lift_start,
                                         int passenger) {
  message_creator mc;
  Position pick_up{49, 9.0};
  Position drop_off{49, 11.0};

  mc.create_and_finish(
      MsgContent_RidesharingBook,
      CreateRidesharingBook(mc, driver, time_lift_start, passenger, 1, 567,
                            &pick_up, 0, &drop_off, 0, 250)
          .Union(),
      "/ridesharing/book");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_book(int driver, int time_lift_start,
                                         int passenger, geo::latlng const& piu,
                                         geo::latlng const& dro,
                                         uint16_t from_leg, uint16_t to_leg) {
  message_creator mc;
  Position pick_up{piu.lat_, piu.lng_};
  Position drop_off{dro.lat_, dro.lng_};

  mc.create_and_finish(
      MsgContent_RidesharingBook,
      CreateRidesharingBook(mc, driver, time_lift_start, passenger, 1,
                            time_lift_start + 15 * 60, &pick_up, from_leg,
                            &drop_off, to_leg, 250)
          .Union(),
      "/ridesharing/book");
  return make_msg(mc);
}

msg_ptr rs_super_itest::ridesharing_get_lifts(int id) {
  message_creator mc;
  mc.create_and_finish(MsgContent_RidesharingGetLiftsRequest,
                       CreateRidesharingGetLiftsRequest(mc, id).Union(),
                       "/ridesharing/lifts");
  return make_msg(mc);
}

}  // namespace motis::ridesharing
