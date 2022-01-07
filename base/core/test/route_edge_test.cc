#include "gtest/gtest.h"

#include "motis/core/schedule/edges.h"

using namespace motis;

static_light_connection make_static_lcon(mam_t const dep, mam_t const arr) {
  static auto const always = bitfield{"1"};
  auto lcon = static_light_connection{dep, arr};
  lcon.traffic_days_ = &always;
  return lcon;
}

auto const static_e =
    make_static_route_edge(nullptr, nullptr,
                           {make_static_lcon(0, 10), make_static_lcon(1, 11),
                            make_static_lcon(2, 12)});

auto const rt_e = make_rt_route_edge(
    nullptr, nullptr,
    {rt_light_connection(motis::time{0, 0}, motis::time{0, 10}),
     rt_light_connection(motis::time{0, 1}, motis::time{0, 11}),
     rt_light_connection(motis::time{0, 2}, motis::time{0, 12})});

TEST(core_route_edge, get_connection_test_valid) {
  auto const c = static_e.get_connection(motis::time{0, 1});

  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ((motis::time{0, 1}), c.d_time());
  ASSERT_EQ((motis::time{0, 11}), c.a_time());
}

TEST(core_route_edge, get_connection_test_valid_last) {
  EXPECT_TRUE(nullptr == static_e.get_connection(motis::time{0, 3}));
}

TEST(core_route_edge, get_connection_invalid_next_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[1].valid_ = 0U;

  auto const c = e1.get_connection(motis::time{0, 1});
  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ((motis::time{0, 2}), c.d_time());
  ASSERT_EQ((motis::time{0, 12}), c.a_time());
}

TEST(core_route_edge, get_connection_2_invalid_next_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[0].valid_ = 0U;
  e1.rt_lcons()[1].valid_ = 0U;

  auto const c = e1.get_connection(motis::time{0, 0});
  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ((motis::time{0, 2}), c.d_time());
  ASSERT_EQ((motis::time{0, 12}), c.a_time());
}

TEST(core_route_edge, get_connection_end_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[1].valid_ = 0U;
  e1.rt_lcons()[2].valid_ = 0U;

  EXPECT_TRUE(nullptr == e1.get_connection(motis::time{0, 1}));
}

TEST(core_route_edge, get_connection_reverse_valid_1_test) {
  auto const c = static_e.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ((motis::time{0, 2}), c.d_time());
  EXPECT_EQ((motis::time{0, 12}), c.a_time());
}

TEST(core_route_edge, get_connection_reverse_valid_2_test) {
  auto const c = static_e.get_connection<search_dir::BWD>(motis::time{0, 11});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ((motis::time{0, 1}), c.d_time());
  EXPECT_EQ((motis::time{0, 11}), c.a_time());
}

TEST(core_route_edge, get_connection_reverse_invalid_next_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[2].valid_ = 0U;

  auto const c = e1.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ((motis::time{0, 1}), c.d_time());
  EXPECT_EQ((motis::time{0, 11}), c.a_time());
}

TEST(core_route_edge, get_connection_reverse_2_invalid_next_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[1].valid_ = 0U;
  e1.rt_lcons()[2].valid_ = 0U;

  auto const c = e1.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ((motis::time{0, 0}), c.d_time());
  EXPECT_EQ((motis::time{0, 10}), c.a_time());
}

TEST(core_route_edge, get_connection_reverse_end_test) {
  auto e1 = rt_e;
  e1.rt_lcons()[0].valid_ = 0U;
  e1.rt_lcons()[1].valid_ = 0U;

  EXPECT_TRUE(nullptr ==
              e1.get_connection<search_dir::BWD>(motis::time{0, 11}));
}
