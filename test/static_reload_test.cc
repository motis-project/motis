#include "gtest/gtest.h"

#include "motis/static_reload.h"

using namespace motis;
using namespace std::chrono;

namespace {

config::timetable timetable_with(
    std::map<std::string, std::optional<std::string>> const& cron_by_tag) {
  auto t = config::timetable{};
  for (auto const& [tag, cron] : cron_by_tag) {
    t.datasets_.emplace(tag,
                        config::timetable::dataset{.path_ = tag + ".gtfs.zip",
                                                   .reload_cron_ = cron});
  }
  return t;
}

}  // namespace

TEST(motis, due_static_reloads_none_configured) {
  auto const t = timetable_with({{"de", std::nullopt}});
  auto const due = due_static_reloads(t, {}, system_clock::now());
  EXPECT_TRUE(due.empty());
}

TEST(motis, due_static_reloads_not_yet_due) {
  // daily at 03:30; "last_fired" defaults to `now` when absent from the map
  // -> not due immediately after (re-)starting.
  auto const t = timetable_with({{"idfm", "0 30 3 * * *"}});
  auto const now = system_clock::now();
  EXPECT_TRUE(due_static_reloads(t, {}, now).empty());
}

TEST(motis, due_static_reloads_due_after_scheduled_time_passed) {
  auto const t = timetable_with({{"idfm", "0 30 3 * * *"}});

  // pretend it last fired 2 days ago -- the schedule has definitely elapsed
  // at least once since then, so it must be considered due now.
  auto const last_fired = std::map<std::string, system_clock::time_point>{
      {"idfm", system_clock::now() - hours{48}}};
  auto const due = due_static_reloads(t, last_fired, system_clock::now());
  ASSERT_EQ(1U, due.size());
  EXPECT_EQ("idfm", due.at(0));
}

TEST(motis, due_static_reloads_multiple_datasets_independent) {
  auto const t = timetable_with(
      {{"idfm", "0 30 3 * * *"}, {"de", "0 0 4 * * *"}, {"nl", std::nullopt}});

  auto const now = system_clock::now();
  auto const last_fired = std::map<std::string, system_clock::time_point>{
      {"idfm", now - hours{48}},  // due
      {"de", now}  // not due yet
  };

  auto const due = due_static_reloads(t, last_fired, now);
  ASSERT_EQ(1U, due.size());
  EXPECT_EQ("idfm", due.at(0));
}

TEST(motis, due_static_reloads_invalid_cron_is_skipped_not_thrown) {
  auto const t = timetable_with({{"idfm", "not a cron expression"}});
  // must not throw -- an invalid expression here should only ever be
  // possible if config::verify() was bypassed; due_static_reloads degrades
  // to "never due" for that dataset rather than crashing the reload loop.
  EXPECT_NO_THROW({
    auto const due = due_static_reloads(t, {}, system_clock::now());
    EXPECT_TRUE(due.empty());
  });
}
