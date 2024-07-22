#include "gtest/gtest.h"

#include "utl/to_vec.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"

using namespace motis;
using namespace motis::module;
using namespace flatbuffers;
using routing::RoutingResponse;

msg_ptr journeys_to_message(std::vector<journey> const& journeys) {
  message_creator fbb;
  std::vector<Offset<Statistics>> s{};
  fbb.create_and_finish(
      MsgContent_RoutingResponse,
      routing::CreateRoutingResponse(
          fbb, fbb.CreateVectorOfSortedTables(&s),
          fbb.CreateVector(utl::to_vec(
              journeys, [&](auto&& j) { return to_connection(fbb, j); })),
          0, 0, fbb.CreateVector(std::vector<Offset<DirectConnection>>{}))
          .Union());
  return make_msg(fbb);
}

journey create_journey1() {
  journey j;
  j.duration_ = 30;
  j.price_ = 10;
  j.transfers_ = 2;
  j.db_costs_ = 100;
  j.night_penalty_ = 200;

  j.stops_.resize(4);
  {
    auto& stop = j.stops_[0];
    stop.eva_no_ = "1111111";
    stop.enter_ = true;
    stop.exit_ = false;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop0";
    stop.arrival_.valid_ = false;
    stop.departure_.valid_ = true;
    stop.departure_.timestamp_ = 1445261400;
    stop.departure_.track_ = "1";
  }
  {
    auto& stop = j.stops_[1];
    stop.eva_no_ = "2222222";
    stop.enter_ = true;
    stop.exit_ = true;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop1";
    stop.arrival_.valid_ = true;
    stop.arrival_.timestamp_ = 1445262000;
    stop.arrival_.track_ = "2";
    stop.departure_.valid_ = true;
    stop.departure_.timestamp_ = 1445262240;
    stop.departure_.track_ = "3";
  }
  {
    auto& stop = j.stops_[2];
    stop.eva_no_ = "3333333";
    stop.enter_ = true;
    stop.exit_ = true;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop2";
    stop.arrival_.valid_ = true;
    stop.arrival_.timestamp_ = 1445262900;
    stop.arrival_.track_ = "4";
    stop.departure_.valid_ = true;
    stop.departure_.timestamp_ = 1445262900;
    stop.departure_.track_ = "";
  }
  {
    auto& stop = j.stops_[3];
    stop.eva_no_ = "4444444";
    stop.enter_ = false;
    stop.exit_ = true;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop3";
    stop.arrival_.valid_ = true;
    stop.arrival_.timestamp_ = 1445263200;
    stop.arrival_.track_ = "";
    stop.departure_.valid_ = false;
  }

  j.transports_.resize(3);
  {
    auto& transport = j.transports_[0];
    transport.direction_ = "X";
    transport.duration_ = 10;
    transport.from_ = 0;
    transport.line_identifier_ = "l1";
    transport.name_ = "ICE 111";
    transport.provider_ = "DB1";
    transport.provider_url_ = "https://example.com";
    transport.mumo_id_ = 0;
    transport.to_ = 1;
    transport.is_walk_ = false;
  }
  {
    auto& transport = j.transports_[1];
    transport.direction_ = "Y";
    transport.duration_ = 11;
    transport.from_ = 1;
    transport.line_identifier_ = "l2";
    transport.name_ = "IC 222";
    transport.provider_ = "DB2";
    transport.provider_url_ = "https://example.org";
    transport.mumo_id_ = 0;
    transport.to_ = 2;
    transport.is_walk_ = false;
  }
  {
    auto& transport = j.transports_[2];
    transport.is_walk_ = true;
    transport.duration_ = 5;
    transport.from_ = 2;
    transport.to_ = 3;
    transport.direction_ = "";
    transport.line_identifier_ = "";
    transport.name_ = "";
    transport.provider_ = "";
    transport.provider_url_ = "";
    transport.mumo_id_ = 0;
  }

  j.trips_.resize(3);
  {
    auto& trip = j.trips_[0];
    trip.from_ = 0;
    trip.to_ = 2;
    trip.extern_trip_.station_id_ = "S";
    trip.extern_trip_.train_nr_ = 1;
    trip.extern_trip_.time_ = 1445261200;
    trip.extern_trip_.target_station_id_ = "T";
    trip.extern_trip_.target_time_ = 1445231200;
    trip.extern_trip_.line_id_ = "1234";
  }
  {
    auto& trip = j.trips_[1];
    trip.from_ = 0;
    trip.to_ = 2;
    trip.extern_trip_.station_id_ = "X";
    trip.extern_trip_.train_nr_ = 2;
    trip.extern_trip_.time_ = 1445261201;
    trip.extern_trip_.target_station_id_ = "Y";
    trip.extern_trip_.target_time_ = 1445231202;
    trip.extern_trip_.line_id_ = "4321";
  }
  {
    auto& trip = j.trips_[2];
    trip.from_ = 1;
    trip.to_ = 2;
    trip.extern_trip_.station_id_ = "A";
    trip.extern_trip_.train_nr_ = 3;
    trip.extern_trip_.time_ = 1445261203;
    trip.extern_trip_.target_station_id_ = "B";
    trip.extern_trip_.target_time_ = 1445231204;
    trip.extern_trip_.line_id_ = "0";
  }

  j.attributes_.resize(2);
  {
    auto& attribute = j.attributes_[0];
    attribute.from_ = 0;
    attribute.to_ = 1;
    attribute.attr_.text_ = "AAA";
    attribute.attr_.code_ = "A";
  }
  {
    auto& attribute = j.attributes_[1];
    attribute.from_ = 1;
    attribute.to_ = 2;
    attribute.attr_.text_ = "BBB";
    attribute.attr_.code_ = "B";
  }
  return j;
}

journey create_journey2() {
  journey j;
  j.duration_ = 15;
  j.price_ = 10;
  j.transfers_ = 0;
  j.db_costs_ = 100;
  j.night_penalty_ = 200;

  j.stops_.resize(2);
  {
    auto& stop = j.stops_[0];
    stop.eva_no_ = "1111111";
    stop.enter_ = true;
    stop.exit_ = false;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop0";
    stop.arrival_.valid_ = false;
    stop.departure_.valid_ = true;
    stop.departure_.timestamp_ = 1445328000;
    stop.departure_.track_ = "1";
  }
  {
    auto& stop = j.stops_[1];
    stop.eva_no_ = "2222222";
    stop.enter_ = false;
    stop.exit_ = true;
    stop.lat_ = 0.0;
    stop.lng_ = 0.0;
    stop.name_ = "Stop1";
    stop.arrival_.valid_ = true;
    stop.arrival_.timestamp_ = 1445328900;
    stop.arrival_.track_ = "2";
    stop.departure_.valid_ = false;
    stop.departure_.timestamp_ = 0;
    stop.departure_.track_ = "3";
  }
  j.transports_.resize(1);
  {
    auto& transport = j.transports_[0];
    transport.direction_ = "X";
    transport.duration_ = 15;
    transport.from_ = 0;
    transport.line_identifier_ = "l1";
    transport.name_ = "ICE 111";
    transport.provider_ = "DB1";
    transport.provider_url_ = "https://www.example.com";
    transport.mumo_id_ = 0;
    transport.to_ = 1;
    transport.is_walk_ = false;
  }
  j.trips_.resize(1);
  {
    auto& trip = j.trips_[0];
    trip.from_ = 0;
    trip.to_ = 2;
    trip.extern_trip_.station_id_ = "S";
    trip.extern_trip_.train_nr_ = 1;
    trip.extern_trip_.time_ = 1445261200;
    trip.extern_trip_.target_station_id_ = "T";
    trip.extern_trip_.target_time_ = 1445231200;
    trip.extern_trip_.line_id_ = "1234";
  }
  return j;
}

TEST(core_convert_journey, journey_message_journey) {
  std::vector<journey> original_journeys = {create_journey1(),
                                            create_journey2()};

  auto msg = journeys_to_message(original_journeys);
  auto journeys = message_to_journeys(motis_content(RoutingResponse, msg));

  ASSERT_TRUE(journeys.size() == 2);

  for (auto i = 0UL; i < 2; ++i) {
    auto const& j = journeys[i];
    auto const& o = original_journeys[i];

    ASSERT_EQ(o.duration_, j.duration_);
    EXPECT_EQ(o.transfers_, j.transfers_);
    EXPECT_EQ(o.stops_.size(), j.stops_.size());
    EXPECT_EQ(o.transports_.size(), j.transports_.size());
    EXPECT_EQ(o.attributes_.size(), j.attributes_.size());

    for (auto s = 0UL; s < o.stops_.size(); ++s) {
      auto const& os = o.stops_[s];
      auto const& js = j.stops_[s];
      ASSERT_EQ(os.eva_no_, js.eva_no_);
      ASSERT_EQ(os.enter_, js.enter_);
      ASSERT_EQ(os.exit_, js.exit_);
      ASSERT_EQ(os.lat_, js.lat_);
      ASSERT_EQ(os.lng_, js.lng_);
      ASSERT_EQ(os.name_, js.name_);
      ASSERT_EQ(os.arrival_.valid_, js.arrival_.valid_);
      ASSERT_EQ(os.departure_.valid_, js.departure_.valid_);
      if (os.arrival_.valid_) {
        ASSERT_EQ(os.arrival_.track_, js.arrival_.track_);
        ASSERT_EQ(os.arrival_.timestamp_, js.arrival_.timestamp_);
        ASSERT_EQ(os.arrival_.valid_, js.arrival_.valid_);
      }
      if (os.departure_.valid_) {
        ASSERT_EQ(os.departure_.track_, js.departure_.track_);
        ASSERT_EQ(os.departure_.timestamp_, js.departure_.timestamp_);
        ASSERT_EQ(os.departure_.valid_, js.departure_.valid_);
      }
    }

    for (auto t = 0UL; t < o.transports_.size(); ++t) {
      auto const& ot = o.transports_[t];
      auto const& jt = j.transports_[t];
      ASSERT_EQ(ot.direction_, jt.direction_);
      ASSERT_EQ(ot.duration_, jt.duration_);
      ASSERT_EQ(ot.from_, jt.from_);
      ASSERT_EQ(ot.line_identifier_, jt.line_identifier_);
      ASSERT_EQ(ot.name_, jt.name_);
      ASSERT_EQ(ot.provider_, jt.provider_);
      ASSERT_EQ(ot.provider_url_, jt.provider_url_);
      ASSERT_EQ(ot.mumo_id_, jt.mumo_id_);
      ASSERT_EQ(ot.to_, jt.to_);
      ASSERT_EQ(ot.is_walk_, jt.is_walk_);
      ASSERT_EQ(ot.mumo_price_, jt.mumo_price_);
      ASSERT_EQ(ot.mumo_type_, jt.mumo_type_);
    }

    for (auto s = 0UL; s < o.trips_.size(); ++s) {
      auto const& ot = o.trips_[s];
      auto const& jt = j.trips_[s];
      ASSERT_EQ(ot.from_, jt.from_);
      ASSERT_EQ(ot.to_, jt.to_);
      ASSERT_EQ(ot.extern_trip_.station_id_, jt.extern_trip_.station_id_);
      ASSERT_EQ(ot.extern_trip_.train_nr_, jt.extern_trip_.train_nr_);
      ASSERT_EQ(ot.extern_trip_.time_, jt.extern_trip_.time_);
      ASSERT_EQ(ot.extern_trip_.target_station_id_,
                jt.extern_trip_.target_station_id_);
      ASSERT_EQ(ot.extern_trip_.target_time_, jt.extern_trip_.target_time_);
      ASSERT_EQ(ot.extern_trip_.line_id_, jt.extern_trip_.line_id_);
    }

    for (auto a = 0UL; a < o.attributes_.size(); ++a) {
      auto const& oa = o.attributes_[a];
      auto const& ja = j.attributes_[a];
      ASSERT_EQ(oa.attr_.code_, ja.attr_.code_);
      ASSERT_EQ(oa.attr_.text_, ja.attr_.text_);
      ASSERT_EQ(oa.from_, ja.from_);
      ASSERT_EQ(oa.to_, ja.to_);
    }
  }
}
