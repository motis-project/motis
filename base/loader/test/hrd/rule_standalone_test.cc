#include "gtest/gtest.h"

#include "motis/loader/hrd/hrd_parser.h"
#include "motis/schedule-format/Schedule_generated.h"

#include "./paths.h"

using namespace motis::loader;
using namespace motis::loader::hrd;
using namespace flatbuffers64;

std::string get_simple_traffic_days(std::string const& t) {
  auto s = t.substr(t.size() - 3);
  std::reverse(begin(s), end(s));
  return s;
}

bool contains_service(Vector<Offset<Service>> const* services, int train_nr,
                      std::string const& traffic_days) {
  return std::any_of(std::begin(*services), std::end(*services), [&](auto&& s) {
    return get_simple_traffic_days(s->traffic_days()->str()) == traffic_days &&
           s->sections()->Get(0)->train_nr() == train_nr;
  });
}

TEST(loader_hrd_fbs_services, rule_standalone) {
  auto const hrd_root = SCHEDULES / "rule-standalone";
  hrd_parser p;
  FlatBufferBuilder b;

  ASSERT_TRUE(p.applicable(hrd_root));

  p.parse(hrd_root, b);
  auto schedule = GetSchedule(b.GetBufferPointer());

  ASSERT_EQ(1U, schedule->rule_services()->size());
  ASSERT_EQ(1U, schedule->rule_services()->Get(0)->rules()->size());

  ASSERT_EQ("010", get_simple_traffic_days(schedule->rule_services()
                                               ->Get(0)
                                               ->rules()
                                               ->Get(0)
                                               ->service1()
                                               ->traffic_days()
                                               ->str()));

  EXPECT_TRUE(contains_service(schedule->services(), 1, "001"));
  EXPECT_TRUE(contains_service(schedule->services(), 2, "100"));
  EXPECT_TRUE(contains_service(schedule->services(), 3, "101"));
}
