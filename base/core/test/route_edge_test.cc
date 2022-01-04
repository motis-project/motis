#include "gtest/gtest.h"

#include "motis/core/schedule/edges.h"

using namespace motis;

light_connection make_lcon(mam_t const dep, mam_t const arr) {
  static auto const always = bitfield{"1"};
  auto lcon = light_connection{dep, arr};
  lcon.traffic_days_ = &always;
  lcon.valid_ = true;
  return lcon;
}

auto const e = make_route_edge(
    nullptr, nullptr, {make_lcon(0, 10), make_lcon(1, 11), make_lcon(2, 12)});

TEST(core_route_edge, get_connection_test_valid) {
  auto [c, day] = e.get_connection(motis::time{0, 1});

  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ(1, c->d_time_);
  ASSERT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_test_valid_last) {
  EXPECT_TRUE(nullptr == e.get_connection(motis::time{0, 3}).first);
}

TEST(core_route_edge, get_connection_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  auto [c, day] = e1.get_connection(motis::time{0, 1});
  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ(2, c->d_time_);
  ASSERT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_2_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[0].valid_ = 0U;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  auto [c, day] = e1.get_connection(motis::time{0, 0});
  ASSERT_TRUE(c != nullptr);
  ASSERT_EQ(2, c->d_time_);
  ASSERT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_end_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  EXPECT_TRUE(nullptr == e1.get_connection(motis::time{0, 1}).first);
}

TEST(core_route_edge, get_connection_reverse_valid_1_test) {
  auto [c, day] = e.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(2, c->d_time_);
  EXPECT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_valid_2_test) {
  auto [c, day] = e.get_connection<search_dir::BWD>(motis::time{0, 11});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(1, c->d_time_);
  EXPECT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  auto [c, day] = e1.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(1, c->d_time_);
  EXPECT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_2_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  auto [c, day] = e1.get_connection<search_dir::BWD>(motis::time{0, 20});
  ASSERT_TRUE(c != nullptr);
  EXPECT_EQ(0, c->d_time_);
  EXPECT_EQ(10, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_end_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[0].valid_ = 0U;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  EXPECT_TRUE(nullptr ==
              e1.get_connection<search_dir::BWD>(motis::time{0, 11}).first);
}