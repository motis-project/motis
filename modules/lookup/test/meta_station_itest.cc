#include "gtest/gtest.h"

#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace motis;
using namespace motis::module;
using namespace motis::test;
using motis::test::schedule::simple_realtime::dataset_opt;

namespace motis::lookup {

constexpr auto kEmptyMetaStationRequest = R""({
  "destination": { "type": "Module", "target": "/lookup/meta_station" },
  "content_type": "LookupMetaStationRequest",
  "content": { "station_id": "8000105" }}
)"";

constexpr auto kMetaStationRequest = R""({
  "destination": { "type": "Module", "target": "/lookup/meta_station" },
  "content_type": "LookupMetaStationRequest",
  "content": { "station_id": "8073368" }}
)"";

struct lookup_meta_station_test : public motis_instance_test {
  lookup_meta_station_test() : motis_instance_test(dataset_opt, {"lookup"}) {}
};

TEST_F(lookup_meta_station_test, no_meta_station) {
  auto const msg = call(make_msg(kEmptyMetaStationRequest));
  auto const resp = motis_content(LookupMetaStationResponse, msg);
  ASSERT_EQ(1, resp->equivalent()->size());
  EXPECT_EQ("8000105", resp->equivalent()->Get(0)->id()->str());
}

TEST_F(lookup_meta_station_test, meta_station) {
  auto const msg = call(make_msg(kMetaStationRequest));
  auto const resp = motis_content(LookupMetaStationResponse, msg);
  ASSERT_EQ(2, resp->equivalent()->size());
  EXPECT_EQ("8003368", resp->equivalent()->Get(0)->id()->str());
  EXPECT_EQ("8073368", resp->equivalent()->Get(1)->id()->str());
}

}  // namespace motis::lookup
