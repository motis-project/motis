#include "gtest/gtest.h"

#include "motis/core/schedule/event_type.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/module/module.h"

#include "motis/revise/update_journey.h"

#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/update_journey.h"

using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::test;
using namespace motis::test::schedule::update_journey;
using motis::test::schedule::update_journey::dataset_opt;

struct revise_itest : public motis_instance_test {
  revise_itest()
      : motis_instance_test(dataset_opt, {"routing", "rt", "ris", "revise"}) {}
};

TEST_F(revise_itest, update_tracks) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  j.stops_.at(0).departure_.schedule_track_ = "10";
  j.stops_.at(0).departure_.track_ = "10";
  j.stops_.at(5).arrival_.schedule_track_ = "20";
  j.stops_.at(5).arrival_.track_ = "15";
  j.stops_.at(10).arrival_.schedule_track_ = "30";
  j.stops_.at(10).arrival_.track_ = "30";
  j.stops_.at(10).departure_.schedule_track_ = "30";
  j.stops_.at(10).departure_.track_ = "30";
  j.stops_.at(13).departure_.schedule_track_ = "40";
  j.stops_.at(13).departure_.track_ = "40";

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_EQ(updated_j.stops_.at(0).departure_.schedule_track_, "1");
  ASSERT_EQ(updated_j.stops_.at(0).departure_.track_, "1");
  ASSERT_EQ(updated_j.stops_.at(5).arrival_.schedule_track_, "1");
  ASSERT_EQ(updated_j.stops_.at(5).arrival_.track_, "1");
  ASSERT_EQ(updated_j.stops_.at(10).arrival_.schedule_track_, "1");
  ASSERT_EQ(updated_j.stops_.at(10).arrival_.track_, "1");
  ASSERT_EQ(updated_j.stops_.at(10).departure_.schedule_track_, "2");
  ASSERT_EQ(updated_j.stops_.at(10).departure_.track_, "2");
  ASSERT_EQ(updated_j.stops_.at(13).arrival_.schedule_track_, "2");
  ASSERT_EQ(updated_j.stops_.at(13).arrival_.track_, "2");
}

TEST_F(revise_itest, update_timestamps) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  j.stops_.at(0).departure_.timestamp_ = unix_time(2200);
  j.stops_.at(5).arrival_.timestamp_ = unix_time(2200);
  j.stops_.at(5).departure_.timestamp_ = unix_time(2200);
  j.stops_.at(10).arrival_.timestamp_ = unix_time(2200);
  j.stops_.at(10).departure_.timestamp_ = unix_time(2200);
  j.stops_.at(13).arrival_.timestamp_ = unix_time(2200);

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_EQ(updated_j.stops_.at(0).departure_.timestamp_, unix_time(1500));
  ASSERT_EQ(updated_j.stops_.at(5).arrival_.timestamp_, unix_time(1512));
  ASSERT_EQ(updated_j.stops_.at(5).departure_.timestamp_, unix_time(1512));
  ASSERT_EQ(updated_j.stops_.at(10).arrival_.timestamp_, unix_time(1525));
  ASSERT_EQ(updated_j.stops_.at(10).departure_.timestamp_, unix_time(1537));
  ASSERT_EQ(updated_j.stops_.at(13).arrival_.timestamp_, unix_time(1612));
}

TEST_F(revise_itest, update_free_texts) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  publish(get_free_text_ris_msg("8000068", 35345, event_type::ARR,
                                unix_time(1525), "8002059", unix_time(1500), 36,
                                "technische Störung", "Extern"));
  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_EQ(updated_j.free_texts_.size(), 1);
  ASSERT_EQ(updated_j.free_texts_.at(0).text_.text_, "technische Störung");
  ASSERT_EQ(updated_j.free_texts_.at(0).text_.type_, "Extern");
  ASSERT_EQ(updated_j.free_texts_.at(0).text_.code_, 36);
  ASSERT_EQ(updated_j.free_texts_.at(0).from_, 10);
  ASSERT_EQ(updated_j.free_texts_.at(0).to_, 10);
}

TEST_F(revise_itest, update_status_ok_without_walk) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());
  publish(get_delay_ris_msg("8000068", 35345, event_type::ARR, unix_time(1525),
                            unix_time(1532), "8002059", unix_time(1500)));
  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;
  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_TRUE(updated_j1.problems_.empty());
}

TEST_F(revise_itest, update_status_ok_with_walk) {
  auto const con = call((get_routing_request(unix_time(1500), unix_time(1612),
                                             "8002042", "8000156")));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());
  publish(get_delay_ris_msg("8098105", 35547, event_type::ARR, unix_time(1506),
                            unix_time(1510), "8002042", unix_time(1500)));
  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;
  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_TRUE(updated_j1.problems_.empty());
}

TEST_F(revise_itest, update_status_violated_interchange_time_with_walk) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002042", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());
  publish(get_delay_ris_msg("8098105", 35547, event_type::ARR, unix_time(1506),
                            unix_time(1511), "8002042", unix_time(1500)));
  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;
  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_EQ(updated_j1.problems_.size(), 1);
  ASSERT_EQ(updated_j1.problems_.at(0).from_, 3);
  ASSERT_EQ(updated_j1.problems_.at(0).to_, 4);
  ASSERT_EQ(updated_j1.problems_.at(0).type_,
            journey::problem_type::INTERCHANGE_TIME_VIOLATED);
}

TEST_F(revise_itest, update_status_violated_interchange_time_without_walk) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());

  publish(get_delay_ris_msg("8000068", 35345, event_type::ARR, unix_time(1525),
                            unix_time(1533), "8002059", unix_time(1500)));
  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;
  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_EQ(updated_j1.problems_.size(), 1);
  ASSERT_EQ(updated_j1.problems_.at(0).from_, 10);
  ASSERT_EQ(updated_j1.problems_.at(0).to_, 10);
  ASSERT_EQ(updated_j1.problems_.at(0).type_,
            journey::problem_type::INTERCHANGE_TIME_VIOLATED);
}

TEST_F(revise_itest,
       DISABLED_update_status_violated_canceled_train_without_walk) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002059", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());

  publish(get_canceled_train_ris_message(
      "8000068", 35345, event_type::ARR, unix_time(1525), "8001378",
      event_type::DEP, unix_time(1522), "8002059", unix_time(1500)));

  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;

  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_EQ(updated_j1.problems_.size(), 1);
  ASSERT_EQ(updated_j1.problems_.at(0).from_, 0);
  ASSERT_EQ(updated_j1.problems_.at(0).to_, 10);
  ASSERT_EQ(updated_j1.problems_.at(0).type_,
            journey::problem_type::CANCELED_TRAIN);
}

TEST_F(revise_itest, DISABLED_update_status_violated_canceled_train_with_walk) {
  auto const con = call(get_routing_request(unix_time(1500), unix_time(1612),
                                            "8002042", "8000156"));
  auto const resp_routing = motis_content(RoutingResponse, con);
  auto journeys = message_to_journeys(resp_routing);

  ASSERT_EQ(journeys.size(), 1);
  auto j = journeys.front();

  message_creator fbb;
  fbb.create_and_finish(MsgContent_Connection, to_connection(fbb, j).Union(),
                        "/revise", DestinationType_Topic);
  auto const resp_update = call(make_msg(fbb));
  auto const updated_connection = motis_content(Connection, resp_update);
  auto updated_journeys = convert(updated_connection);

  auto updated_j = updated_journeys;

  ASSERT_TRUE(updated_j.problems_.empty());

  publish(get_canceled_train_ris_message(
      "8098105", 35547, event_type::ARR, unix_time(1506), "8006690",
      event_type::DEP, unix_time(1504), "8002042", unix_time(1500)));

  publish(make_no_msg("/ris/system_time_changed"));

  message_creator fbb1;
  fbb1.create_and_finish(MsgContent_Connection, to_connection(fbb1, j).Union(),
                         "/revise", DestinationType_Topic);
  auto const resp_update1 = call(make_msg(fbb1));
  auto const updated_connection1 = motis_content(Connection, resp_update1);
  auto updated_journeys1 = convert(updated_connection1);

  auto updated_j1 = updated_journeys1;

  ASSERT_EQ(updated_j1.problems_.size(), 1);
  ASSERT_EQ(updated_j1.problems_.at(0).from_, 0);
  ASSERT_EQ(updated_j1.problems_.at(0).to_, 3);
  ASSERT_EQ(updated_j1.problems_.at(0).type_,
            journey::problem_type::CANCELED_TRAIN);
}
