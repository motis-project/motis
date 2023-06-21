#include "gtest/gtest.h"

#include "nigiri/loader/gtfs/files.h"
#include "nigiri/loader/gtfs/load_timetable.h"
#include "nigiri/loader/init_finish.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/timetable.h"

using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;

namespace {

// ROUTING CONNECTIONS:
// 10:00 - 11:00, 1h, 1 transfer
// START -> [TRIP 1: A, B, C] -> WALK [5 min] -> [TRIP 2: D, E] -> DEST
//
// 10:00 - 10:45, 45min, 2 transfers
// START -> [TRIP 1: A, B] -> [TRIP 3: B, C] -> [TRIP 4: C, E] -> DEST
//
// TIMETABLE + DELAYS:
// TRIP  |  STOP  |  EV  |  PLANNED  | REAL TIME
// ------+--------+------+-----------+-------------
// 1     |  A     | DEP  |  09:50    | 10:00 (+10)
// 1     |  B     | ARR  |  10:00    | 10:10 (+10)
// 1     |  B     | DEP  |  10:10    | 10:15 (+5)
// 1     |  C     | ARR  |  10:30    | 10:35 (+5)
// 2     |  D     | DEP  |  10:25    | 10:40 (+15)
// 2     |  E     | ARR  |  10:45    | 11:00 (+15)
// 3     |  B     | DEP  |  10:10    | 10:20 (+10)
// 3     |  C     | ARR  |  10:30    | 10:30 (+0)
// 4     |  C     | DEP  |  10:30    | 10:35 (+5)
// 4     |  E     | ARR  |  10:45    | 10:45 (+0)
n::loader::mem_dir test_files() {
  using std::filesystem::path;
  return {
      {{path{n::loader::gtfs::kAgencyFile},
        std::string{
            R"(agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin
)"}},
       {path{n::loader::gtfs::kStopFile},
        std::string{
            R"(stop_id,stop_name,stop_desc,stop_lat,stop_lon,stop_url,location_type,parent_station
A,A,,0.0,1.0,,
B,B,,2.0,3.0,,
C,C,,4.0,5.0,,
D,D,,6.0,7.0,,
E,E,,7.0,8.0,,
)"}},
       {path{n::loader::gtfs::kCalendarDatesFile},
        std::string{R"(service_id,date,exception_type
S1,20190501,1
)"}},
       {path{n::loader::gtfs::kRoutesFile},
        std::string{
            R"(route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,1,,,3
R2,DB,2,,,3
R3,DB,3,,,3
R4,DB,4,,,3
)"}},
       {path{n::loader::gtfs::kTripsFile},
        std::string{R"(route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,
R2,S1,T2,,
R3,S1,T3,,
R4,S1,T4,,
)"}},
       {path{n::loader::gtfs::kStopTimesFile},
        std::string{
            R"(trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,09:50:00,09:50:00,A,1,0,0
T1,10:00:00,10:10:00,B,2,0,0
T1,10:30:00,10:30:00,C,3,0,0
T2,10:25:00,10:25:00,D,1,0,0
T2,10:45:00,10:45:00,E,2,0,0
T3,10:10:00,10:10:00,B,1,0,0
T3,10:30:00,10:30:00,C,2,0,0
T4,10:30:00,10:30:00,C,1,0,0
T4,10:45:00,10:45:00,E,2,0,0
)"}},
       {path{n::loader::gtfs::kTransfersFile},
        std::string{R"(from_stop_id,to_stop_id,transfer_type,min_transfer_time
C,D,0,2
)"}}}};
}

}  // namespace

struct trip {
  struct delay {
    std::string stop_id_;
    unsigned delay_minutes_{0U};
  };
  std::string trip_id_;
  std::vector<delay> delays_;
};

template <typename T>
std::int64_t to_unix(T&& x) {
  return std::chrono::time_point_cast<std::chrono::seconds>(x)
      .time_since_epoch()
      .count();
};

transit_realtime::FeedMessage to_feed_msg(std::vector<trip> const& trip_delays,
                                          date::sys_seconds const msg_time) {
  transit_realtime::FeedMessage msg;

  auto const hdr = msg.mutable_header();
  hdr->set_gtfs_realtime_version("2.0");
  hdr->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  hdr->set_timestamp(to_unix(msg_time));

  auto id = 0U;
  for (auto const& trip : trip_delays) {
    auto const e = msg.add_entity();
    e->set_id(fmt::format("{}", ++id));
    e->set_is_deleted(false);

    auto const td = e->mutable_trip_update()->mutable_trip();
    td->set_trip_id(trip.trip_id_);

    for (auto const& stop_delay : trip.delays_) {
      auto* const upd = e->mutable_trip_update()->add_stop_time_update();
      *upd->mutable_stop_id() = stop_delay.stop_id_;
      upd->mutable_departure()->set_delay(stop_delay.delay_minutes_ * 60);
    }
  }

  return msg;
}

TEST(nigiri, rt_test) {
  auto tt = n::timetable{};
  tt.date_range_ = {date::sys_days{2019_y / March / 25},
                    date::sys_days{2019_y / November / 1}};
  n::loader::gtfs::load_timetable({}, n::source_idx_t{0}, test_files(), tt);
  n::loader::finalize(tt);

  auto rtt = n::rt::create_rt_timetable(tt, date::sys_days{2019_y / May / 1});

  auto const stats = n::rt::gtfsrt_update_msg(
      tt, rtt, n::source_idx_t{0}, "tag",
      to_feed_msg(
          {trip{.trip_id_ = "T1",
                .delays_ = {trip::delay{.stop_id_ = "A", .delay_minutes_ = 10},
                            trip::delay{.stop_id_ = "B", .delay_minutes_ = 5}}},
           trip{
               .trip_id_ = "T2",
               .delays_ = {trip::delay{.stop_id_ = "D", .delay_minutes_ = 15}}},
           trip{.trip_id_ = "T3",
                .delays_ = {trip::delay{.stop_id_ = "B", .delay_minutes_ = 10},
                            trip::delay{.stop_id_ = "C", .delay_minutes_ = 0}}},
           trip{
               .trip_id_ = "T4",
               .delays_ = {trip::delay{.stop_id_ = "C", .delay_minutes_ = 5},
                           trip::delay{.stop_id_ = "E", .delay_minutes_ = 0}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(stats.total_entities_success_, 4U);
}