#include "gtest/gtest.h"
#include "motis/transport_mode_ids.h"

#include "motis/odm/get_td_offsets.h"

using namespace nigiri;
using namespace nigiri::routing;
using namespace std::chrono_literals;

namespace motis::odm {

TEST(odm, get_td_offsets_basic) {
  auto const rides = std::vector<start>{{.time_at_start_ = unixtime_t{10h},
                                         .time_at_stop_ = unixtime_t{11h},
                                         .stop_ = location_idx_t{1U}}};

  auto const td_offsets =
      motis::odm::get_td_offsets(rides, kOdmTransportModeId);

  ASSERT_TRUE(td_offsets.contains(location_idx_t{1U}));
  ASSERT_EQ(td_offsets.at(location_idx_t{1U}).size(), 2U);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].valid_from_, unixtime_t{10h});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].duration_, 1h);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].transport_mode_id_,
            kOdmTransportModeId);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].valid_from_,
            unixtime_t{10h + 1min});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].duration_,
            footpath::kMaxDuration);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].transport_mode_id_,
            kOdmTransportModeId);
}

TEST(odm, get_td_offsets_extension) {
  auto const rides =
      std::vector<start>{{.time_at_start_ = unixtime_t{10h},
                          .time_at_stop_ = unixtime_t{11h},
                          .stop_ = location_idx_t{1U}},
                         {.time_at_start_ = unixtime_t{10h + 1min},
                          .time_at_stop_ = unixtime_t{11h + 1min},
                          .stop_ = location_idx_t{1U}},
                         {.time_at_start_ = unixtime_t{10h + 2min},
                          .time_at_stop_ = unixtime_t{11h + 2min},
                          .stop_ = location_idx_t{1U}}};

  auto const td_offsets =
      motis::odm::get_td_offsets(rides, kOdmTransportModeId);

  ASSERT_TRUE(td_offsets.contains(location_idx_t{1U}));
  ASSERT_EQ(td_offsets.at(location_idx_t{1U}).size(), 2U);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].valid_from_, unixtime_t{10h});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].duration_, 1h);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].transport_mode_id_,
            kOdmTransportModeId);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].valid_from_,
            unixtime_t{10h + 3min});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].duration_,
            footpath::kMaxDuration);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].transport_mode_id_,
            kOdmTransportModeId);
}

TEST(odm, get_td_offsets_intermittent) {
  auto const rides = std::vector<start>{{.time_at_start_ = unixtime_t{10h},
                                         .time_at_stop_ = unixtime_t{11h},
                                         .stop_ = location_idx_t{1U}},
                                        {.time_at_start_ = unixtime_t{11h},
                                         .time_at_stop_ = unixtime_t{12h},
                                         .stop_ = location_idx_t{1U}},
                                        {.time_at_start_ = unixtime_t{12h},
                                         .time_at_stop_ = unixtime_t{13h},
                                         .stop_ = location_idx_t{1U}}};

  auto const td_offsets =
      motis::odm::get_td_offsets(rides, kOdmTransportModeId);

  ASSERT_TRUE(td_offsets.contains(location_idx_t{1U}));
  ASSERT_EQ(td_offsets.at(location_idx_t{1U}).size(), 6U);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].valid_from_, unixtime_t{10h});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].duration_, 1h);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[0].transport_mode_id_,
            kOdmTransportModeId);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].valid_from_,
            unixtime_t{10h + 1min});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].duration_,
            footpath::kMaxDuration);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[1].transport_mode_id_,
            kOdmTransportModeId);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[2].valid_from_, unixtime_t{11h});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[2].duration_, 1h);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[2].transport_mode_id_,
            kOdmTransportModeId);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[3].valid_from_,
            unixtime_t{11h + 1min});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[3].duration_,
            footpath::kMaxDuration);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[3].transport_mode_id_,
            kOdmTransportModeId);

  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[4].valid_from_, unixtime_t{12h});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[4].duration_, 1h);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[4].transport_mode_id_,
            kOdmTransportModeId);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[5].valid_from_,
            unixtime_t{12h + 1min});
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[5].duration_,
            footpath::kMaxDuration);
  EXPECT_EQ(td_offsets.at(location_idx_t{1U})[5].transport_mode_id_,
            kOdmTransportModeId);
}

}  // namespace motis::odm