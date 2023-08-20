#include "gtest/gtest.h"

#include <string>

#include "fmt/format.h"

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"

#include "geo/latlng.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/core/journey/print_journey.h"
#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace geo;
using namespace flatbuffers;
using namespace motis::osrm;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::intermodal {

namespace {
loader::loader_options add_tag(loader::loader_options opt) {
  opt.dataset_prefix_.emplace_back("x");
  return opt;
}
}  // namespace

struct intermodal_itest
    : public generic_motis_instance_test<testing::TestWithParam<const char*>> {
  intermodal_itest()
      : motis::test::generic_motis_instance_test<
            testing::TestWithParam<const char*>>(
            add_tag(dataset_opt),
            {"intermodal", "routing", "tripbased", "nigiri", "lookup"},
            {"--tripbased.use_data_file=false", "--nigiri.lookup=false",
             "--nigiri.routing=false", "--nigiri.first_day=2015-11-24"}) {
    instance_->register_op(
        "/osrm/one_to_many",
        [](msg_ptr const& msg) {
          auto const req = motis_content(OSRMOneToManyRequest, msg);
          auto const one = latlng{req->one()->lat(), req->one()->lng()};
          auto const speed =
              req->profile()->str() == "bike" ? BIKE_SPEED : WALK_SPEED;

          std::vector<Cost> costs;
          for (auto const& loc : *req->many()) {
            auto dist = distance(one, {loc->lat(), loc->lng()});
            costs.emplace_back(dist / speed, dist);
          }

          message_creator mc;
          mc.create_and_finish(
              MsgContent_OSRMOneToManyResponse,
              CreateOSRMOneToManyResponse(mc, mc.CreateVectorOfStructs(costs))
                  .Union());
          return make_msg(mc);
        },
        {});
  }
};

TEST_F(intermodal_itest, forward) {
  //  Heidelberg Hbf -> Bensheim ( departure: 2015-11-24 13:30:00 )
  auto json = [](std::string_view router) {
    return fmt::format(R"({{
      "destination": {{
        "type": "Module",
        "target": "/intermodal"
      }},
      "content_type": "IntermodalRoutingRequest",
      "content": {{
        "start_type": "IntermodalOntripStart",
        "start": {{
          "position": {{ "lat": 49.4047178, "lng": 8.6768716}},
          "departure_time": 1448368200
        }},
        "start_modes": [{{
          "mode_type": "Foot",
          "mode": {{ "max_duration": 600 }}
        }}],
        "destination_type": "InputPosition",
        "destination": {{ "lat": 49.6801332, "lng": 8.6200666}},
        "destination_modes":  [{{
          "mode_type": "Foot",
          "mode": {{ "max_duration": 600 }}
        }},{{
          "mode_type": "Bike",
          "mode": {{ "max_duration": 600 }}
        }}],
        "search_type": "Default",
        "router": "{}"
      }}
    }})",
                       router);
  };

  for (auto const& router : {"/routing", "/tripbased", "/nigiri"}) {
    SCOPED_TRACE(router);

    auto res = call(make_msg(json(router)));
    auto content = motis_content(RoutingResponse, res);

    ASSERT_EQ(1, content->connections()->size());

    print_journey(message_to_journeys(content)[0], std::cout);

    auto const& stops = content->connections()->Get(0)->stops();

    ASSERT_EQ(5, stops->size());

    auto const& start = stops->Get(0);
    EXPECT_STREQ(STATION_START, start->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.4047178, start->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.6768716, start->station()->pos()->lng());

    auto const& first_station = stops->Get(1);
    EXPECT_STREQ("x_8000156", first_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

    auto const& last_station = stops->Get(3);
    EXPECT_STREQ("x_8000031", last_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());

    auto const& end = stops->Get(4);
    EXPECT_STREQ(STATION_END, end->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.6801332, end->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.6200666, end->station()->pos()->lng());

    auto const& transports = content->connections()->Get(0)->transports();
    ASSERT_EQ(3, transports->size());

    ASSERT_EQ(Move_Walk, transports->Get(0)->move_type());
    ASSERT_STREQ(
        "foot", reinterpret_cast<motis::Walk const*>(transports->Get(0)->move())
                    ->mumo_type()
                    ->c_str());

    ASSERT_EQ(Move_Walk, transports->Get(0)->move_type());
    ASSERT_STREQ(
        "bike", reinterpret_cast<motis::Walk const*>(transports->Get(2)->move())
                    ->mumo_type()
                    ->c_str());
  }
}

TEST_F(intermodal_itest, backward) {
  //  Heidelberg Hbf -> Bensheim ( arrival: 2015-11-24 14:30:00 )
  auto json = [](std::string_view router) {
    return fmt::format(R"({{
      "destination": {{
        "type": "Module",
        "target": "/intermodal"
      }},
      "content_type": "IntermodalRoutingRequest",
      "content": {{
        "start_type": "IntermodalOntripStart",
        "start": {{
          "position": {{ "lat": 49.6801332, "lng": 8.6200666 }},
          "departure_time": 1448371800
        }},
        "start_modes": [{{
          "mode_type": "Foot",
          "mode": {{ "max_duration": 600 }}
        }}],
        "destination_type": "InputPosition",
        "destination": {{ "lat": 49.4047178, "lng": 8.6768716 }},
        "destination_modes":  [{{
          "mode_type": "Foot",
          "mode": {{ "max_duration": 600 }}
        }}],
        "search_type": "Default",
        "search_dir": "Backward",
        "router": "{}"
      }}
    }}
  )",
                       router);
  };

  for (auto const& router : {"/routing", "/tripbased", "/nigiri"}) {
    SCOPED_TRACE(router);

    auto res = call(make_msg(json(router)));
    auto content = motis_content(RoutingResponse, res);

    ASSERT_EQ(1, content->connections()->size());

    print_journey(message_to_journeys(content)[0], std::cout);

    auto const& stops = content->connections()->Get(0)->stops();

    ASSERT_EQ(5, stops->size());

    auto const& start = stops->Get(0);
    EXPECT_STREQ(STATION_END, start->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.4047178, start->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.6768716, start->station()->pos()->lng());

    auto const& first_station = stops->Get(1);
    EXPECT_STREQ("x_8000156", first_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

    auto const& last_station = stops->Get(3);
    EXPECT_STREQ("x_8000031", last_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());

    auto const& end = stops->Get(4);
    EXPECT_STREQ(STATION_START, end->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.6801332, end->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.6200666, end->station()->pos()->lng());
  }
}

TEST_F(intermodal_itest, not_so_intermodal) {
  //  Heidelberg Hbf -> Bensheim ( departure: 2015-11-24 13:30:00 )
  auto json = [](std::string_view router) {
    return fmt::format(R"(
    {{
      "destination": {{
        "type": "Module",
        "target": "/intermodal"
      }},
      "content_type": "IntermodalRoutingRequest",
      "content": {{
        "start_type": "OntripStationStart",
        "start": {{
          "station": {{ "id": "x_8000156", "name": "" }},
          "departure_time": 1448368200
        }},
        "start_modes": [],
        "destination_type": "InputStation",
        "destination": {{ "id": "x_8000031", "name": "" }},
        "destination_modes": [],
        "search_type": "Default",
        "router": ""
      }}
    }}
  )",
                       router);
  };

  for (auto const& router : {"/routing", "/tripbased", "/nigiri"}) {
    auto res = call(make_msg(json(router)));
    auto content = motis_content(RoutingResponse, res);

    ASSERT_EQ(1, content->connections()->size());
    auto const& stops = content->connections()->Get(0)->stops();

    ASSERT_EQ(3, stops->size());

    auto const& first_station = stops->Get(0);
    EXPECT_STREQ("x_8000156", first_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.403567, first_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.675442, first_station->station()->pos()->lng());

    auto const& last_station = stops->Get(2);
    EXPECT_STREQ("x_8000031", last_station->station()->id()->c_str());
    EXPECT_DOUBLE_EQ(49.681329, last_station->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(8.616717, last_station->station()->pos()->lng());
  }
}

}  // namespace motis::intermodal
