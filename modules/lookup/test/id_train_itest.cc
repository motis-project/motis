#include "gtest/gtest.h"

#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace motis;
using namespace motis::module;
using namespace motis::test;
using namespace motis::lookup;
using motis::test::schedule::simple_realtime::dataset_opt;

constexpr auto kIdTrainICERequest = R""(
{
  "destination": {
    "type": "Module",
    "target": "/lookup/id_train"
  },
  "content_type": "LookupIdTrainRequest",
  "content": {
    "trip_id": {
      "id": "",
      "station_id": "8000261",
      "train_nr": 628,
      "time": 1448362440,
      "target_station_id": "8000080",
      "target_time": 1448382360,
      "line_id": ""
    }
  }
}
)"";

struct lookup_id_train_test : public motis_instance_test {
  lookup_id_train_test()
      : motis_instance_test(
            dataset_opt, {"lookup", "rt"},
            {"--ris.input=test/schedule/simple_realtime/risml/delays.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}
};

// TODO(sebastian) re-enable when working realtime module is available
TEST_F(lookup_id_train_test, DISABLED_id_train) {
  auto msg = call(make_msg(kIdTrainICERequest));
  auto resp = motis_content(LookupIdTrainResponse, msg);

  auto stops = resp->train()->stops();
  ASSERT_EQ(12, stops->size());

  {
    auto s = stops->Get(0);
    EXPECT_EQ("8000261", s->station()->id()->str());
    EXPECT_EQ("München Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(48.140232, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(11.558335, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(0, a->schedule_time());
    EXPECT_EQ(0, a->time());
    EXPECT_EQ("", a->track()->str());

    auto d = s->departure();
    EXPECT_EQ(1448362440, d->schedule_time());
    EXPECT_EQ(1448362440, d->time());
    EXPECT_EQ("23", d->track()->str());
  }
  {
    auto s = stops->Get(3);
    EXPECT_EQ("8000010", s->station()->id()->str());
    EXPECT_EQ("Aschaffenburg Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(49.980557, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(9.143697, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(1448372040, a->schedule_time());
    EXPECT_EQ(1448372040, a->time());
    EXPECT_EQ("8", a->track()->str());

    auto d = s->departure();
    EXPECT_EQ(1448372160, d->schedule_time());
    EXPECT_EQ(1448372220, d->time());  // +1
    EXPECT_EQ("8", d->track()->str());
  }
  {
    auto s = stops->Get(11);
    EXPECT_EQ("8000080", s->station()->id()->str());
    EXPECT_EQ("Dortmund Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(51.517896, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(7.459290, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(1448382360, a->schedule_time());
    EXPECT_EQ(1448382600, a->time());  // +4 (assuming min standing time = 2)
    EXPECT_EQ("", a->track()->str());  // unknown

    auto d = s->departure();
    EXPECT_EQ(0, d->schedule_time());
    EXPECT_EQ(0, d->time());
    EXPECT_EQ("", d->track()->str());
  }
}

TEST_F(lookup_id_train_test, no_realtime) {
  auto msg = call(make_msg(kIdTrainICERequest));
  auto resp = motis_content(LookupIdTrainResponse, msg);

  auto stops = resp->train()->stops();
  ASSERT_EQ(12, stops->size());

  {
    auto s = stops->Get(0);
    EXPECT_EQ("8000261", s->station()->id()->str());
    EXPECT_EQ("München Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(48.140232, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(11.558335, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(0, a->schedule_time());
    EXPECT_EQ(0, a->time());
    EXPECT_EQ("", a->track()->str());

    auto d = s->departure();
    EXPECT_EQ(1448362440, d->schedule_time());
    EXPECT_EQ(1448362440, d->time());
    EXPECT_EQ("23", d->track()->str());
  }
  {
    auto s = stops->Get(3);
    EXPECT_EQ("8000010", s->station()->id()->str());
    EXPECT_EQ("Aschaffenburg Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(49.980557, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(9.143697, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(1448372040, a->schedule_time());
    EXPECT_EQ(1448372040, a->time());
    EXPECT_EQ("8", a->track()->str());

    auto d = s->departure();
    EXPECT_EQ(1448372160, d->schedule_time());
    EXPECT_EQ(1448372160, d->time());  // +1
    EXPECT_EQ("8", d->track()->str());
  }
  {
    auto s = stops->Get(11);
    EXPECT_EQ("8000080", s->station()->id()->str());
    EXPECT_EQ("Dortmund Hbf", s->station()->name()->str());
    EXPECT_DOUBLE_EQ(51.517896, s->station()->pos()->lat());
    EXPECT_DOUBLE_EQ(7.459290, s->station()->pos()->lng());

    auto a = s->arrival();
    EXPECT_EQ(1448382360, a->schedule_time());
    EXPECT_EQ(1448382360, a->time());
    EXPECT_EQ("", a->track()->str());  // unknown

    auto d = s->departure();
    EXPECT_EQ(0, d->schedule_time());
    EXPECT_EQ(0, d->time());
    EXPECT_EQ("", d->track()->str());
  }
}
