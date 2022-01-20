#include "boost/geometry.hpp"

#include "motis/core/common/logging.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"

#include "motis/module/message.h"
#include "motis/module/module.h"

#include "motis/protocol/Message_generated.h"
#include "motis/ridesharing/lift.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

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
using namespace motis::intermodal;
using motis::logging::info;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::ridesharing {

struct rs_intermodal_test : public motis_instance_test {
  rs_intermodal_test()
      : motis::test::motis_instance_test(
            dataset_opt, {"lookup", "ridesharing", "intermodal", "routing"},
            {"--ridesharing.database_path=:memory:",
             "--ridesharing.use_parking=false"}) {
    instance_->register_op("/osrm/many_to_many", [&](msg_ptr const& msg) {
      auto const req = motis_content(OSRMManyToManyRequest, msg);
      std::vector<geo::latlng> coor;
      for (auto const& loc : *req->coordinates()) {
        coor.emplace_back(latlng{loc->lat(), loc->lng()});
      }

      message_creator mc;
      std::vector<Offset<OSRMManyToManyResponseRow>> routing_matrix;
      for (auto const& from : coor) {
        std::vector<Cost> cost_row;
        for (auto const& to : coor) {
          auto dist = distance(from, to);
          if (req->profile()->str() == "car") {
            cost_row.emplace_back(dist / CAR_SPEED, dist);
          } else {
            cost_row.emplace_back(dist / WALK_SPEED, dist);
          }
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
      auto const req = motis_content(OSRMOneToManyRequest, msg);
      auto one = geo::latlng{req->one()->lat(), req->one()->lng()};

      std::vector<Cost> costs;
      for (auto const& loc : *req->many()) {
        auto dist = distance(one, {loc->lat(), loc->lng()});
        if (req->profile()->str() == "car") {
          costs.emplace_back(dist / CAR_SPEED, dist);
        } else {
          costs.emplace_back(dist / WALK_SPEED, dist);
        }
      }

      message_creator mc;
      mc.create_and_finish(
          MsgContent_OSRMOneToManyResponse,
          CreateOSRMOneToManyResponse(mc, mc.CreateVectorOfStructs(costs))
              .Union());
      return make_msg(mc);
    });

    instance_->register_op("/osrm/route", [&](msg_ptr const& msg) {
      auto const req = motis_content(OSRMRouteRequest, msg);
      auto const coordinates = utl::to_vec(*req->coordinates(), [](auto* co) {
        return geo::latlng{co->lat(), co->lng()};
      });
      std::vector<Cost> costs;
      for (auto i = 0u; i < coordinates.size() - 1; i++) {
        auto dist = distance(coordinates[i], coordinates[i + 1]);
        if (req->profile()->str() == "car") {
          costs.emplace_back(dist / CAR_SPEED, dist);
        } else {
          costs.emplace_back(dist / WALK_SPEED, dist);
        }
      }

      message_creator mc;
      mc.create_and_finish(
          MsgContent_OSRMRouteResponse,
          CreateOSRMRouteResponse(mc, mc.CreateVectorOfStructs(costs)).Union());
      return make_msg(mc);
    });
  }
};

TEST_F(rs_intermodal_test, intermodal_ridesharing_entry) {
  //  Heidelberg Hbf -> Bensheim ( arrival: 2015-11-24 14:30:00 )

  publish(make_no_msg("/osrm/initialized"));
  message_creator mc;

  Position stuttgart{48.784, 9.182};
  Position heidelberg{49.405, 8.677};
  Position bensheim{49.681, 8.6167};

  mc.create_and_finish(MsgContent_RidesharingCreate,
                       CreateRidesharingCreate(mc, 12345, unix_time(1020), 4,
                                               &stuttgart, &heidelberg)
                           .Union(),
                       "/ridesharing/create");
  call(make_msg(mc));

  mc.create_and_finish(MsgContent_RidesharingCreate,
                       CreateRidesharingCreate(mc, 12345, unix_time(1420), 4,
                                               &bensheim, &heidelberg)
                           .Union(),
                       "/ridesharing/create");
  call(make_msg(mc));

  Position query_start{stuttgart.lat() - 0.001,
                       stuttgart.lng() - 0.001};  // close to stuttgart
  auto const dest_loc = geo::latlng{
      bensheim.lat() - 0.001, bensheim.lng() - 0.001};  // close to bensheim
  std::vector<Offset<ModeWrapper>> start_modes{CreateModeWrapper(
      mc, Mode_Ridesharing,
      CreateRidesharing(mc, 2,
                        motis::ppr::CreateSearchOptions(
                            mc, mc.CreateString("default"), 100000))
          .Union())};
  std::vector<Offset<ModeWrapper>> end_modes{
      CreateModeWrapper(mc, Mode_Foot, CreateFoot(mc, 600).Union())};
  mc.create_and_finish(
      MsgContent_IntermodalRoutingRequest,
      CreateIntermodalRoutingRequest(
          mc, IntermodalStart_IntermodalOntripStart,
          CreateIntermodalOntripStart(mc, &query_start, 1448355840).Union(),
          mc.CreateVector(start_modes), IntermodalDestination_InputPosition,
          CreateInputPosition(mc, dest_loc.lat_, dest_loc.lng_).Union(),
          mc.CreateVector(end_modes), SearchType_DefaultPrice,
          SearchDir_Forward)
          .Union(),
      "/intermodal");

  auto const res = call(make_msg(mc));
  auto const content = motis_content(RoutingResponse, res);
  auto const journeys = message_to_journeys(content);

  ASSERT_EQ(1, journeys.size());

  auto const rj = journeys[0];
  auto const stops = rj.stops_;
  auto const transports = rj.transports_;

  ASSERT_EQ(5, stops.size());
  ASSERT_EQ(3, transports.size());
  auto const start = rj.stops_[0];
  auto const station_stop = rj.stops_[1];
  auto const dest = rj.stops_[4];
  auto const t = rj.transports_[0];
  ASSERT_EQ("START", start.eva_no_);
  ASSERT_EQ("8000156", station_stop.eva_no_);
  ASSERT_EQ("END", dest.eva_no_);
  EXPECT_FLOAT_EQ(query_start.lat(), start.lat_);
  EXPECT_FLOAT_EQ(query_start.lng(), start.lng_);
  EXPECT_FLOAT_EQ(dest_loc.lat_, dest.lat_);
  EXPECT_FLOAT_EQ(dest_loc.lng_, dest.lng_);
  ASSERT_EQ(1032, t.mumo_price_);
  ASSERT_EQ(72, t.duration_);
  ASSERT_EQ(1, t.is_walk_);
  ASSERT_EQ("ridesharing", t.mumo_type_);
  ASSERT_EQ("1448356800;12345", t.provider_);
  ASSERT_EQ(0, t.from_leg_);
  ASSERT_EQ(0, t.to_leg_);

  ASSERT_EQ(0, rj.accessibility_);
  ASSERT_EQ(1032, rj.price_);
  ASSERT_EQ(250, rj.duration_);

  auto direct = content->direct_connections();
  ASSERT_EQ(0, direct->size());
}

TEST_F(rs_intermodal_test, immediate_exit) {
  publish(make_no_msg("/osrm/initialized"));
  message_creator mc;

  Position stuttgart{48.784, 9.182};
  Position bensheim{49.681, 8.6167};

  mc.create_and_finish(MsgContent_RidesharingCreate,
                       CreateRidesharingCreate(mc, 12345, unix_time(1020), 4,
                                               &stuttgart, &bensheim)
                           .Union(),
                       "/ridesharing/create");
  call(make_msg(mc));

  Position pick_up{stuttgart.lat(), stuttgart.lng() - 0.001};
  Position drop_off{stuttgart.lat(), stuttgart.lng() + 0.001};

  mc.create_and_finish(MsgContent_RidesharingBook,
                       CreateRidesharingBook(mc, 12345, unix_time(1020), 123, 1,
                                             unix_time(1020) + 10 * 60,
                                             &pick_up, 0, &drop_off, 0, 250)
                           .Union(),
                       "/ridesharing/book");
  call(make_msg(mc));

  Position query_start{stuttgart.lat() - 0.001, stuttgart.lng() - 0.001};
  auto const dest_loc =
      geo::latlng{bensheim.lat() - 0.001, bensheim.lng() - 0.001};
  std::vector<Offset<ModeWrapper>> start_modes{CreateModeWrapper(
      mc, Mode_Ridesharing,
      CreateRidesharing(mc, 2,
                        motis::ppr::CreateSearchOptions(
                            mc, mc.CreateString("default"), 100000))
          .Union())};
  std::vector<Offset<ModeWrapper>> end_modes{
      CreateModeWrapper(mc, Mode_Foot, CreateFoot(mc, 600).Union())};
  mc.create_and_finish(
      MsgContent_IntermodalRoutingRequest,
      CreateIntermodalRoutingRequest(
          mc, IntermodalStart_IntermodalOntripStart,
          CreateIntermodalOntripStart(mc, &query_start, 1448355840).Union(),
          mc.CreateVector(start_modes), IntermodalDestination_InputPosition,
          CreateInputPosition(mc, dest_loc.lat_, dest_loc.lng_).Union(),
          mc.CreateVector(end_modes), SearchType_DefaultPrice,
          SearchDir_Forward)
          .Union(),
      "/intermodal");

  auto const res = call(make_msg(mc));
  auto const content = motis_content(RoutingResponse, res);
  auto const journeys = message_to_journeys(content);

  ASSERT_EQ(1, journeys.size());

  auto const rj = journeys[0];
  auto const stops = rj.stops_;
  auto const transports = rj.transports_;

  auto const start = rj.stops_[0];
  auto const station_stop = rj.stops_[1];
  auto const dest = rj.stops_[2];
  auto const t = rj.transports_[0];
  ASSERT_EQ("START", start.eva_no_);
  ASSERT_EQ("8000031", station_stop.eva_no_);
  ASSERT_EQ("END", dest.eva_no_);
  ASSERT_DOUBLE_EQ(query_start.lat(), start.lat_);
  ASSERT_DOUBLE_EQ(query_start.lng(), start.lng_);
  ASSERT_DOUBLE_EQ(dest_loc.lat_, dest.lat_);
  ASSERT_DOUBLE_EQ(dest_loc.lng_, dest.lng_);
  ASSERT_EQ(648, t.mumo_price_);
  ASSERT_EQ(93, t.duration_);
  ASSERT_EQ(1, t.is_walk_);
  ASSERT_EQ("ridesharing", t.mumo_type_);
  ASSERT_EQ("1448356800;12345", t.provider_);
  ASSERT_EQ(1, t.from_leg_);
  ASSERT_EQ(2, t.to_leg_);

  ASSERT_EQ(45, rj.accessibility_);
  ASSERT_EQ(648, rj.price_);
  ASSERT_EQ(94, rj.duration_);

  auto direct = content->direct_connections();
  ASSERT_EQ(0, direct->size());
}

TEST_F(rs_intermodal_test, direct_connection) {
  publish(make_no_msg("/osrm/initialized"));
  message_creator mc;

  Position stuttgart{48.784, 9.182};
  Position bensheim{49.681, 8.6167};

  mc.create_and_finish(MsgContent_RidesharingCreate,
                       CreateRidesharingCreate(mc, 12345, unix_time(1020), 4,
                                               &stuttgart, &bensheim)
                           .Union(),
                       "/ridesharing/create");
  call(make_msg(mc));

  Position query_start{stuttgart.lat() - 0.001, stuttgart.lng() - 0.001};
  auto const dest_loc =
      geo::latlng{bensheim.lat() - 0.001, bensheim.lng() - 0.001};
  std::vector<Offset<ModeWrapper>> start_modes{CreateModeWrapper(
      mc, Mode_Ridesharing,
      CreateRidesharing(mc, 2,
                        motis::ppr::CreateSearchOptions(
                            mc, mc.CreateString("default"), 100000))
          .Union())};
  std::vector<Offset<ModeWrapper>> end_modes{CreateModeWrapper(
      mc, Mode_Ridesharing,
      CreateRidesharing(mc, 2,
                        motis::ppr::CreateSearchOptions(
                            mc, mc.CreateString("default"), 100000))
          .Union())};
  mc.create_and_finish(
      MsgContent_IntermodalRoutingRequest,
      CreateIntermodalRoutingRequest(
          mc, IntermodalStart_IntermodalOntripStart,
          CreateIntermodalOntripStart(mc, &query_start, unix_time(1004))
              .Union(),
          mc.CreateVector(start_modes), IntermodalDestination_InputPosition,
          CreateInputPosition(mc, dest_loc.lat_, dest_loc.lng_).Union(),
          mc.CreateVector(end_modes), SearchType_DefaultPrice,
          SearchDir_Forward)
          .Union(),
      "/intermodal");

  auto const res = call(make_msg(mc));
  auto const content = motis_content(RoutingResponse, res);
  auto const journeys = message_to_journeys(content);

  ASSERT_EQ(1, journeys.size());
  auto const rj = journeys[0];
  auto const start = rj.stops_[0];
  auto const dest = rj.stops_[1];
  auto const t = rj.transports_[0];
  ASSERT_EQ("START", start.eva_no_);
  ASSERT_EQ("END", dest.eva_no_);
  ASSERT_DOUBLE_EQ(query_start.lat(), start.lat_);
  ASSERT_DOUBLE_EQ(query_start.lng(), start.lng_);
  ASSERT_DOUBLE_EQ(dest_loc.lat_, dest.lat_);
  ASSERT_DOUBLE_EQ(dest_loc.lng_, dest.lng_);
  ASSERT_EQ(648, t.mumo_price_);
  ASSERT_EQ(80, t.duration_);
  ASSERT_EQ(1, t.is_walk_);
  ASSERT_EQ("ridesharing", t.mumo_type_);
  ASSERT_EQ("1448356800;12345", t.provider_);
  ASSERT_EQ(0, t.from_leg_);
  ASSERT_EQ(0, t.to_leg_);

  ASSERT_EQ(0, rj.accessibility_);
  ASSERT_EQ(648, rj.price_);
  ASSERT_EQ(80, rj.duration_);

  auto direct = content->direct_connections();
  ASSERT_EQ(1, direct->size());
  auto dc = direct->Get(0);
  ASSERT_EQ(64, dc->duration());
  ASSERT_EQ(0, dc->accessibility());
  EXPECT_STREQ("ridesharing", dc->mumo_type()->c_str());
  ASSERT_EQ(648, dc->price());
  ASSERT_EQ(1448356805, dc->departure_time());
}

}  // namespace motis::ridesharing
