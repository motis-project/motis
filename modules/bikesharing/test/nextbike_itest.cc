#include "gtest/gtest.h"

#include <iostream>

#include "motis/module/message.h"

#include "motis/test/motis_instance_test.h"

using namespace motis::test;
using namespace motis::module;

namespace motis::bikesharing {

constexpr auto kBikesharingRequestDeparture = R""(
{
  "destination": {
    "type": "Module",
    "target": "/bikesharing/search"
  },
  "content_type": "BikesharingRequest",
  "content": {
    "type": Departure,

    // close to campus darmstadt
    "position": {
      "lat": 49.8776114,
      "lng": 8.6571044
    },

    "interval": {
      "begin": 1454602500,  // Thu, 04 Feb 2016 16:15:00 GMT
      "end": 1454606100  // Thu, 04 Feb 2016 17:15:00 GMT
    },

    "availability_aggregator": "Average"
  }
}
)"";

constexpr auto kBikesharingRequestArrival = R""(
{
  "destination": {
    "type": "Module",
    "target": "/bikesharing/search"
  },
  "content_type": "BikesharingRequest",
  "content": {
    "type": Arrival,

    // close to campus ffm
    "position": {
      "lat": 50.1273104,
      "lng": 8.6669383
    },

    "interval": {
      "begin": 1454602500,  // Thu, 04 Feb 2016 16:15:00 GMT
      "end": 1454606100  // Thu, 04 Feb 2016 17:15:00 GMT
    },

    "availability_aggregator": "Average"
  }
}
)"";

// nextbike-1454603400.xml -> Thu, 04 Feb 2016 16:30:00 GMT
// nextbike-1454603700.xml -> Thu, 04 Feb 2016 16:35:00 GMT

template <typename V>
BikesharingEdge const* find_edge(V const* vec, std::string const& from,
                                 std::string const& to) {
  for (auto e : *vec) {
    if (e->from()->id()->str() == from && e->to()->id()->str() == to) {
      return e;
    }
  }
  return nullptr;
}

class bikesharing_nextbike_itest : public test::motis_instance_test {
public:
  bikesharing_nextbike_itest()
      : test::motis_instance_test(
            {{"modules/bikesharing/test_resources/schedule"},
             "20150112",
             2,
             false,
             false,
             false,
             true},
            {"lookup", "bikesharing"},
            {"--bikesharing.nextbike_path=modules/"
             "bikesharing/test_resources/nextbike",
             "--bikesharing.database_path=:memory:"}) {}
};

TEST_F(bikesharing_nextbike_itest, integration_test_departure) {
  auto msg = call(make_msg(kBikesharingRequestDeparture));

  ASSERT_EQ(MsgContent_BikesharingResponse, msg->get()->content_type());
  using bikesharing::BikesharingResponse;
  auto resp = motis_content(BikesharingResponse, msg);

  /****************************************************************************
   *  check departure side
   ****************************************************************************/
  {
    auto dep_edges = resp->edges();
    ASSERT_EQ(4, dep_edges->size());

    auto e_1_3 = find_edge(dep_edges, "1", "3");
    ASSERT_NE(nullptr, e_1_3);
    EXPECT_EQ(std::string("8000068"), e_1_3->station_id()->str());
    ASSERT_EQ(2, e_1_3->availability()->size());

    auto e_1_a0 = e_1_3->availability()->Get(0);
    EXPECT_EQ(1454601600, e_1_a0->begin());  // Thu, 04 Feb 2016 16:00:00 GMT
    EXPECT_EQ(1454605200, e_1_a0->end());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(3, e_1_a0->value());

    auto e_1_a1 = e_1_3->availability()->Get(1);
    EXPECT_EQ(1454605200, e_1_a1->begin());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1454608800, e_1_a1->end());  // Thu, 04 Feb 2016 18:00:00 GMT
    EXPECT_EQ(0, e_1_a1->value());

    auto e_1_4 = find_edge(dep_edges, "1", "4");
    ASSERT_NE(nullptr, e_1_4);
    EXPECT_EQ(std::string("8000068"), e_1_4->station_id()->str());
    ASSERT_EQ(2, e_1_4->availability()->size());

    auto e_2_3 = find_edge(dep_edges, "2", "3");
    ASSERT_NE(nullptr, e_2_3);
    EXPECT_EQ(std::string("8000068"), e_2_3->station_id()->str());
    ASSERT_EQ(2, e_2_3->availability()->size());

    auto e_2_a0 = e_2_3->availability()->Get(0);
    EXPECT_EQ(1454601600, e_2_a0->begin());  // Thu, 04 Feb 2016 16:00:00 GMT
    EXPECT_EQ(1454605200, e_2_a0->end());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(4.5, e_2_a0->value());

    auto e_2_a1 = e_2_3->availability()->Get(1);
    EXPECT_EQ(1454605200, e_2_a1->begin());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1454608800, e_2_a1->end());  // Thu, 04 Feb 2016 18:00:00 GMT
    EXPECT_EQ(0, e_2_a1->value());

    auto e_2_4 = find_edge(dep_edges, "2", "4");
    ASSERT_NE(nullptr, e_2_4);
    EXPECT_EQ(std::string("8000068"), e_2_4->station_id()->str());
    ASSERT_EQ(2, e_2_4->availability()->size());
  }
}

TEST_F(bikesharing_nextbike_itest, integration_test_arrival) {
  auto msg = call(make_msg(kBikesharingRequestArrival));

  ASSERT_EQ(MsgContent_BikesharingResponse, msg->get()->content_type());
  using bikesharing::BikesharingResponse;
  auto resp = motis_content(BikesharingResponse, msg);

  /****************************************************************************
   *  check arrival side
   ****************************************************************************/
  {
    auto arr_edges = resp->edges();
    ASSERT_EQ(4, arr_edges->size());

    auto e_1_3 = find_edge(arr_edges, "7", "5");
    ASSERT_NE(nullptr, e_1_3);
    EXPECT_EQ(std::string("8000105"), e_1_3->station_id()->str());
    ASSERT_EQ(26, e_1_3->availability()->size());

    auto e_1_a0 = e_1_3->availability()->Get(0);
    EXPECT_EQ(1454601600, e_1_a0->begin());  // Thu, 04 Feb 2016 16:00:00 GMT
    EXPECT_EQ(1454605200, e_1_a0->end());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1, e_1_a0->value());

    auto e_1_a1 = e_1_3->availability()->Get(1);
    EXPECT_EQ(1454605200, e_1_a1->begin());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1454608800, e_1_a1->end());  // Thu, 04 Feb 2016 18:00:00 GMT
    EXPECT_EQ(0, e_1_a1->value());

    auto e_1_4 = find_edge(arr_edges, "7", "6");
    ASSERT_NE(nullptr, e_1_4);
    EXPECT_EQ(std::string("8000105"), e_1_4->station_id()->str());
    ASSERT_EQ(26, e_1_4->availability()->size());

    auto e_2_3 = find_edge(arr_edges, "8", "5");
    ASSERT_NE(nullptr, e_2_3);
    EXPECT_EQ(std::string("8000105"), e_2_3->station_id()->str());
    ASSERT_EQ(26, e_2_3->availability()->size());

    auto e_2_a0 = e_2_3->availability()->Get(0);
    EXPECT_EQ(1454601600, e_2_a0->begin());  // Thu, 04 Feb 2016 16:00:00 GMT
    EXPECT_EQ(1454605200, e_2_a0->end());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1, e_2_a0->value());

    auto e_2_a1 = e_2_3->availability()->Get(1);
    EXPECT_EQ(1454605200, e_2_a1->begin());  // Thu, 04 Feb 2016 17:00:00 GMT
    EXPECT_EQ(1454608800, e_2_a1->end());  // Thu, 04 Feb 2016 18:00:00 GMT
    EXPECT_EQ(0, e_2_a1->value());

    auto e_2_4 = find_edge(arr_edges, "8", "6");
    ASSERT_NE(nullptr, e_2_4);
    EXPECT_EQ(std::string("8000105"), e_2_4->station_id()->str());
    ASSERT_EQ(26, e_2_4->availability()->size());
  }
}

constexpr auto kBikesharingGeoTerminalRequest = R""(
{
  "destination": {
    "type": "Module",
    "target": "/bikesharing/geo_terminals"
  },
  "content_type": "BikesharingGeoTerminalsRequest",
  "content": {
    // close to campus darmstadt
    "pos": {
      "lat": 49.8776114,
      "lng": 8.6571044
    },
    "radius": 1000,
    "timestamp": 1454602500,  // Thu, 04 Feb 2016 16:15:00 GMT
    "availability_aggregator": "Average"
  }
}
)"";

TEST_F(bikesharing_nextbike_itest, geo_terminals) {
  auto msg = call(make_msg(kBikesharingGeoTerminalRequest));

  using bikesharing::BikesharingGeoTerminalsResponse;
  auto resp = motis_content(BikesharingGeoTerminalsResponse, msg);

  ASSERT_EQ(2, resp->terminals()->size());
  for (auto&& terminal : *resp->terminals()) {
    if (terminal->id()->str() == "1") {
      EXPECT_DOUBLE_EQ(49.8780247, terminal->pos()->lat());
      EXPECT_DOUBLE_EQ(8.6545839, terminal->pos()->lng());
      EXPECT_DOUBLE_EQ(3, terminal->availability());
    } else if (terminal->id()->str() == "2") {
      EXPECT_DOUBLE_EQ(49.875768, terminal->pos()->lat());
      EXPECT_DOUBLE_EQ(8.6578002, terminal->pos()->lng());
      EXPECT_DOUBLE_EQ(4.5, terminal->availability());
    } else {
      FAIL() << "unexpected terminal";
    }
  }
}

}  // namespace motis::bikesharing
