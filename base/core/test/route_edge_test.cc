#include "gtest/gtest.h"

#include "motis/core/schedule/edges.h"

using namespace motis;

auto const e = make_route_edge(
    nullptr, nullptr,
    {light_connection(0, 10, nullptr), light_connection(1, 11, nullptr),
     light_connection(2, 12, nullptr)});

TEST(core_route_edge, get_connection_test_valid) {
  auto c = e.get_connection(1);
  ASSERT_TRUE(c);
  ASSERT_EQ(1, c->d_time_);
  ASSERT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_test_valid_last) {
  EXPECT_FALSE(e.get_connection(3));
}

TEST(core_route_edge, get_connection_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  auto c = e1.get_connection(1);
  ASSERT_TRUE(c);
  ASSERT_EQ(2, c->d_time_);
  ASSERT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_2_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[0].valid_ = 0U;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  auto c = e1.get_connection(0);
  ASSERT_TRUE(c);
  ASSERT_EQ(2, c->d_time_);
  ASSERT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_end_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  EXPECT_FALSE(e1.get_connection(1));
}

TEST(core_route_edge, get_connection_reverse_valid_1_test) {
  auto c = e.get_connection<search_dir::BWD>(20);
  ASSERT_TRUE(c);
  EXPECT_EQ(2, c->d_time_);
  EXPECT_EQ(12, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_valid_2_test) {
  auto c = e.get_connection<search_dir::BWD>(11);
  ASSERT_TRUE(c);
  EXPECT_EQ(1, c->d_time_);
  EXPECT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  auto c = e1.get_connection<search_dir::BWD>(20);
  ASSERT_TRUE(c);
  EXPECT_EQ(1, c->d_time_);
  EXPECT_EQ(11, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_2_invalid_next_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;
  e1.m_.route_edge_.conns_[2].valid_ = 0U;

  auto c = e1.get_connection<search_dir::BWD>(20);
  ASSERT_TRUE(c);
  EXPECT_EQ(0, c->d_time_);
  EXPECT_EQ(10, c->a_time_);
}

TEST(core_route_edge, get_connection_reverse_end_test) {
  auto e1 = e;
  e1.m_.route_edge_.conns_[0].valid_ = 0U;
  e1.m_.route_edge_.conns_[1].valid_ = 0U;

  EXPECT_FALSE(e1.get_connection<search_dir::BWD>(11));
}
