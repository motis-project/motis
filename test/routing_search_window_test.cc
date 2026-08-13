#include "gtest/gtest.h"

#include "motis/endpoints/routing.h"

using namespace std::chrono_literals;
using namespace date;
using namespace motis;
using namespace motis::ep;
namespace n = nigiri;

namespace {

constexpr auto const kQueryTime =
    n::unixtime_t{sys_days{2025_y / September / 15} + 8h};

n::timetable make_tt() {
  auto tt = n::timetable{};
  tt.date_range_ = {sys_days{2025_y / September / 1},
                    sys_days{2025_y / October / 1}};
  return tt;
}

n::interval<n::unixtime_t> start_interval(n::timetable const& tt,
                                          std::int64_t const search_window,
                                          bool const arrive_by,
                                          n::unixtime_t const t = kQueryTime) {
  auto q = api::plan_params{};
  q.time_ = openapi::date_time_t{t};
  q.searchWindow_ = search_window;
  q.arriveBy_ = arrive_by;
  return std::get<n::interval<n::unixtime_t>>(
      get_start_time(q, &tt).first.start_time_);
}

}  // namespace

TEST(motis, search_window_depart_after) {
  auto const tt = make_tt();
  EXPECT_EQ((n::interval<n::unixtime_t>{kQueryTime, kQueryTime + 1h}),
            start_interval(tt, 3600, false));
  EXPECT_EQ((n::interval<n::unixtime_t>{kQueryTime, kQueryTime + 15min}),
            start_interval(tt, 900, false));
}

TEST(motis, search_window_arrive_by) {
  auto const tt = make_tt();
  // arriveBy searches backwards from the query time: the window extends into
  // the past, the query time itself is still contained (half-open interval).
  EXPECT_EQ((n::interval<n::unixtime_t>{kQueryTime - 1h, kQueryTime + 1min}),
            start_interval(tt, 3600, true));
  EXPECT_EQ((n::interval<n::unixtime_t>{kQueryTime - 15min, kQueryTime + 1min}),
            start_interval(tt, 900, true));
}

// Regression test: searchWindow used to be negated before being subtracted for
// arriveBy, so the interval ran *forward* from the query time and got smaller
// as the window grew - large windows produced an empty or inverted interval
// and the query silently returned nothing.
TEST(motis, search_window_grows) {
  auto const tt = make_tt();
  for (auto const arrive_by : {false, true}) {
    auto last = n::interval<n::unixtime_t>{kQueryTime, kQueryTime};
    for (auto const window : {900, 3600, 6 * 3600, 12 * 3600}) {
      auto const i = start_interval(tt, window, arrive_by);

      // never empty or inverted
      EXPECT_LT(i.from_, i.to_)
          << "arriveBy=" << arrive_by << ", searchWindow=" << window;

      // the query time is part of the searched interval
      EXPECT_TRUE(i.contains(kQueryTime))
          << "arriveBy=" << arrive_by << ", searchWindow=" << window;

      // the requested window is covered on the correct side of the query time
      EXPECT_EQ(
          arrive_by ? kQueryTime - std::chrono::seconds{window} : kQueryTime,
          i.from_)
          << "arriveBy=" << arrive_by << ", searchWindow=" << window;
      EXPECT_EQ(arrive_by ? kQueryTime + 1min
                          : kQueryTime + std::chrono::seconds{window},
                i.to_)
          << "arriveBy=" << arrive_by << ", searchWindow=" << window;

      // a bigger search window never searches less
      EXPECT_GT(i.size(), last.size())
          << "arriveBy=" << arrive_by << ", searchWindow=" << window;
      last = i;
    }
  }
}

TEST(motis, search_window_clamped_to_timetable) {
  auto const tt = make_tt();
  auto const ext = tt.external_interval();

  // window reaching beyond the end of the timetable
  EXPECT_EQ((n::interval<n::unixtime_t>{ext.to_ - 1h, ext.to_}),
            start_interval(tt, 12 * 3600, false, ext.to_ - 1h));

  // window reaching beyond the start of the timetable
  EXPECT_EQ((n::interval<n::unixtime_t>{ext.from_, ext.from_ + 1min}),
            start_interval(tt, 12 * 3600, true, ext.from_));
}

TEST(motis, search_window_ontrip) {
  auto const tt = make_tt();
  auto q = api::plan_params{};
  q.time_ = openapi::date_time_t{kQueryTime};
  q.searchWindow_ = 3600;
  q.timetableView_ = false;

  auto const [start_time, t] = get_start_time(q, &tt);
  EXPECT_EQ(kQueryTime, std::get<n::unixtime_t>(start_time.start_time_));
  EXPECT_EQ(kQueryTime, t);
}
