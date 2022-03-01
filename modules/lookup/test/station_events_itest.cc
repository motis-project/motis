#include "gtest/gtest.h"

#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace motis;
using namespace motis::module;
using namespace motis::test;
using namespace motis::lookup;
using motis::test::schedule::simple_realtime::dataset_opt;

constexpr auto kNotInPeriod = R""(
{ "destination": {"type": "Module", "target": "/lookup/station_events"},
  "content_type": "LookupStationEventsRequest",
  "content": { "station_id": "foo", "interval": {"begin": 0, "end": 0} }}
)"";

constexpr auto kSiegenEmptyRequest = R""(
{ "destination": {"type": "Module", "target": "/lookup/station_events"},
  "content_type": "LookupStationEventsRequest",
  "content": {
    "station_id": "8000046",  // Siegen Hbf
    "interval": {
      "begin": 1448373600,  // 2015-11-24 15:00:00 GMT+0100
      "end": 1448374200  // 2015-11-24 15:10:00 GMT+0100
    }
  }}
)"";

constexpr auto kSiegenRequest = R""(
{ "destination": {"type": "Module", "target": "/lookup/station_events"},
  "content_type": "LookupStationEventsRequest",
  "content": {
    "station_id": "8000046",  // Siegen Hbf
    "interval": {
      "begin": 1448373600,  // 2015-11-24 15:00:00 GMT+0100
      "end": 1448374260  // 2015-11-24 15:11:00 GMT+0100
    }
  }}
)"";

constexpr auto kFrankfurtRequest = R""(
{ "destination": {"type": "Module", "target": "/lookup/station_events"},
  "content_type": "LookupStationEventsRequest",
  "content": {
    "station_id": "8000105",  // Frankfurt(Main)Hbf
    "interval": {
      "begin": 1448371800,  // 2015-11-24 14:30:00 GMT+0100
      "end": 1448375400  // 2015-11-24 15:30:00 GMT+0100
    }
  }}
)"";

struct lookup_station_events_test : public motis_instance_test {
  lookup_station_events_test()
      : motis_instance_test(
            dataset_opt, {"lookup", "rt"},
            {"--ris.input=test/schedule/simple_realtime/risml/delays.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}
};

// TODO(sebastian) re-enable when working realtime module is available
TEST_F(lookup_station_events_test, DISABLED_station_events) {
  ASSERT_ANY_THROW(call(make_msg(kNotInPeriod)));

  {
    auto msg = call(make_msg(kSiegenEmptyRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);
    ASSERT_EQ(0, resp->events()->size());  // end is exclusive
  }
  {
    auto msg = call(make_msg(kSiegenRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);
    ASSERT_EQ(1, resp->events()->size());

    auto event = resp->events()->Get(0);
    EXPECT_EQ(EventType_DEP, event->type());
    EXPECT_EQ(10958, event->train_nr());
    EXPECT_EQ("", event->line_id()->str());
    EXPECT_EQ(1448374200, event->time());
    EXPECT_EQ(1448374200, event->schedule_time());

    auto trip_ids = event->trip_id();
    ASSERT_EQ(1, trip_ids->size());
    auto trip_id = trip_ids->Get(0);
    EXPECT_EQ("8000046", trip_id->station_id()->str());
    EXPECT_EQ(10958, trip_id->train_nr());
    EXPECT_EQ("", trip_id->line_id()->str());
    EXPECT_EQ(1448374200, trip_id->time());
  }
  {
    auto msg = call(make_msg(kFrankfurtRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);

    ASSERT_EQ(3, resp->events()->size());
    for (auto e : *resp->events()) {
      ASSERT_NE(nullptr, e);
      auto tids = e->trip_id();
      ASSERT_EQ(1, tids->size());
      auto tid = tids->Get(0);

      switch (e->schedule_time()) {
        case 1448372400:
          EXPECT_EQ(EventType_ARR, e->type());
          EXPECT_EQ(2292, e->train_nr());
          EXPECT_EQ("381", e->line_id()->str());
          EXPECT_EQ(1448372400, e->time());
          EXPECT_EQ(1448372400, e->schedule_time());

          EXPECT_EQ("8000096", tid->station_id()->str());
          EXPECT_EQ(2292, tid->train_nr());
          EXPECT_EQ("381", tid->line_id()->str());
          EXPECT_EQ(1448366700, tid->time());
          break;

        case 1448373840:
          EXPECT_EQ(EventType_ARR, e->type());
          EXPECT_EQ(628, e->train_nr());
          EXPECT_EQ("", e->line_id()->str());
          EXPECT_EQ(1448373900, e->time());
          EXPECT_EQ(1448373840, e->schedule_time());

          EXPECT_EQ("8000261", tid->station_id()->str());
          EXPECT_EQ(628, tid->train_nr());
          EXPECT_EQ("", tid->line_id()->str());
          EXPECT_EQ(1448362440, tid->time());
          break;

        case 1448374200:
          EXPECT_EQ(EventType_DEP, e->type());
          EXPECT_EQ(628, e->train_nr());
          EXPECT_EQ("", e->line_id()->str());
          EXPECT_EQ(1448374200, e->time());
          EXPECT_EQ(1448374200, e->schedule_time());

          EXPECT_EQ("8000261", tid->station_id()->str());
          EXPECT_EQ(628, tid->train_nr());
          EXPECT_EQ("", tid->line_id()->str());
          EXPECT_EQ(1448362440, tid->time());
          break;

        default: FAIL() << "unexpected event"; break;
      }
    }
  }
}

TEST_F(lookup_station_events_test, station_events_no_realtime) {
  ASSERT_ANY_THROW(call(make_msg(kNotInPeriod)));

  {
    auto msg = call(make_msg(kSiegenEmptyRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);
    ASSERT_EQ(0, resp->events()->size());  // end is exclusive
  }
  {
    auto msg = call(make_msg(kSiegenRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);
    ASSERT_EQ(1, resp->events()->size());

    auto event = resp->events()->Get(0);
    EXPECT_EQ(EventType_DEP, event->type());
    EXPECT_EQ(10958, event->train_nr());
    EXPECT_EQ("", event->line_id()->str());
    EXPECT_EQ(1448374200, event->time());
    EXPECT_EQ(1448374200, event->schedule_time());

    auto trip_ids = event->trip_id();
    ASSERT_EQ(1, trip_ids->size());
    auto trip_id = trip_ids->Get(0);
    EXPECT_EQ("8000046", trip_id->station_id()->str());
    EXPECT_EQ(10958, trip_id->train_nr());
    EXPECT_EQ("", trip_id->line_id()->str());
    EXPECT_EQ(1448374200, trip_id->time());
  }
  {
    auto msg = call(make_msg(kFrankfurtRequest));
    auto resp = motis_content(LookupStationEventsResponse, msg);

    ASSERT_EQ(3, resp->events()->size());
    for (auto e : *resp->events()) {
      ASSERT_NE(nullptr, e);
      auto tids = e->trip_id();
      ASSERT_EQ(1, tids->size());
      auto tid = tids->Get(0);

      switch (e->schedule_time()) {
        case 1448372400:
          EXPECT_EQ(EventType_ARR, e->type());
          EXPECT_EQ(2292, e->train_nr());
          EXPECT_EQ("381", e->line_id()->str());
          EXPECT_EQ(1448372400, e->time());
          EXPECT_EQ(1448372400, e->schedule_time());

          EXPECT_EQ("8000096", tid->station_id()->str());
          EXPECT_EQ(2292, tid->train_nr());
          EXPECT_EQ("381", tid->line_id()->str());
          EXPECT_EQ(1448366700, tid->time());
          break;

        case 1448373840:
          EXPECT_EQ(EventType_ARR, e->type());
          EXPECT_EQ(628, e->train_nr());
          EXPECT_EQ("", e->line_id()->str());
          EXPECT_EQ(1448373840, e->time());
          EXPECT_EQ(1448373840, e->schedule_time());

          EXPECT_EQ("8000261", tid->station_id()->str());
          EXPECT_EQ(628, tid->train_nr());
          EXPECT_EQ("", tid->line_id()->str());
          EXPECT_EQ(1448362440, tid->time());
          break;

        case 1448374200:
          EXPECT_EQ(EventType_DEP, e->type());
          EXPECT_EQ(628, e->train_nr());
          EXPECT_EQ("", e->line_id()->str());
          EXPECT_EQ(1448374200, e->time());
          EXPECT_EQ(1448374200, e->schedule_time());

          EXPECT_EQ("8000261", tid->station_id()->str());
          EXPECT_EQ(628, tid->train_nr());
          EXPECT_EQ("", tid->line_id()->str());
          EXPECT_EQ(1448362440, tid->time());
          break;

        default: FAIL() << "unexpected event"; break;
      }
    }
  }
}
