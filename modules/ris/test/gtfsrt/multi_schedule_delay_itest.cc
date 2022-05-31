#include "gtest/gtest.h"

#include "motis/core/access/trip_access.h"

#include "./gtfsrt_itest.h"

using namespace motis;
using namespace motis::test;
using namespace motis::loader;
using namespace motis::module;
using namespace motis::ris::gtfsrt;

static auto const gtfsrt_multi_schedule_dataset_opt =
    loader_options{.dataset_ = {"test/schedule/multi_schedule_poznan/sroda",
                                "test/schedule/multi_schedule_poznan/ztm"},
                   .dataset_prefix_ = {"sroda", "ztm"},
                   .schedule_begin_ = "20211222",
                   .num_days_ = 1};

struct ris_gtfsrt_multi_schedule_delay_message_itest_t0 : public gtfsrt_itest {
  ris_gtfsrt_multi_schedule_delay_message_itest_t0()
      : gtfsrt_itest(gtfsrt_multi_schedule_dataset_opt,
                     {"--ris.instant_forward=true",
                      "--ris.gtfsrt.is_addition_skip_allowed=true",
                      "--ris.input=sroda|test/schedule/multi_schedule_poznan/"
                      "gtfsrt/pb/sroda",
                      "--ris.input=ztm|test/schedule/multi_schedule_poznan/"
                      "gtfsrt/pb/ztm"}) {}
};

TEST_F(ris_gtfsrt_multi_schedule_delay_message_itest_t0, simple) {
  auto const trp_sroda = get_gtfs_trip(
      sched(), gtfs_trip_id{"sroda_", "17_tam_11:15:00", std::nullopt});
  auto const evs_sroda = get_trip_event_info(sched(), trp_sroda);
  EXPECT_EQ(unix_to_motistime(sched(), parse_unix_time("2021-12-22 11:15 CET")),
            evs_sroda.at("sroda_3:122:00").dep_);
  EXPECT_EQ(unix_to_motistime(sched(), parse_unix_time("2021-12-22 11:18 CET")),
            evs_sroda.at("sroda_3:123:00").dep_);

  auto const trp_ztm =
      get_gtfs_trip(sched(), gtfs_trip_id{"ztm_", "1_672491^#", std::nullopt});
  auto const evs_ztm = get_trip_event_info(sched(), trp_ztm);
  EXPECT_EQ(unix_to_motistime(sched(), parse_unix_time("2021-12-22 10:50 CET")),
            evs_ztm.at("ztm_1906").dep_);
  EXPECT_EQ(unix_to_motistime(sched(), parse_unix_time("2021-12-22 10:56 CET")),
            evs_ztm.at("ztm_1851").arr_);
}
