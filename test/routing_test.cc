#include "gtest/gtest.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/json.hpp"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/init_from.h"

#include "nigiri/rt/gtfsrt_update.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/endpoints/routing.h"
#include "motis/gbfs/update.h"
#include "motis/import.h"

namespace json = boost::json;
using namespace std::string_view_literals;
using namespace motis;
using namespace date;
using namespace std::chrono_literals;
namespace n = nigiri;

constexpr auto const kFastaJson = R"__(
[
  {
    "description": "FFM HBF zu Gleis 101/102 (S-Bahn)",
    "equipmentnumber" : 10561326,
    "geocoordX" : 8.6628995,
    "geocoordY" : 50.1072933,
    "operatorname" : "DB InfraGO",
    "state" : "ACTIVE",
    "stateExplanation" : "available",
    "stationnumber" : 1866,
    "type" : "ELEVATOR",
    "outOfService": [
      ["2019-05-01T01:30:00Z", "2019-05-01T02:30:00Z"]
    ]
  },
  {
    "description": "FFM HBF zu Gleis 103/104 (S-Bahn)",
    "equipmentnumber": 10561327,
    "geocoordX": 8.6627516,
    "geocoordY": 50.1074549,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1866,
    "type": "ELEVATOR"
  },
  {
    "description": "HAUPTWACHE zu Gleis 2/3 (S-Bahn)",
    "equipmentnumber": 10351032,
    "geocoordX": 8.67818,
    "geocoordY": 50.114046,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1864,
    "type": "ELEVATOR"
  },
  {
    "description": "DA HBF zu Gleis 1",
    "equipmentnumber": 10543458,
    "geocoordX": 8.6303864,
    "geocoordY": 49.8725612,
    "state": "ACTIVE",
    "type": "ELEVATOR"
  },
  {
    "description": "DA HBF zu Gleis 3/4",
    "equipmentnumber": 10543453,
    "geocoordX": 8.6300911,
    "geocoordY": 49.8725678,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 5/6",
    "equipmentnumber": 10543454,
    "geocoordX": 8.6298163,
    "geocoordY": 49.8725555,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 7/8",
    "equipmentnumber": 10543455,
    "geocoordX": 8.6295535,
    "geocoordY": 49.87254,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 9/10",
    "equipmentnumber": 10543456,
    "geocoordX": 8.6293117,
    "geocoordY": 49.8725263,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  },
  {
    "description": "zu Gleis 11/12",
    "equipmentnumber": 10543457,
    "geocoordX": 8.6290451,
    "geocoordY": 49.8725147,
    "operatorname": "DB InfraGO",
    "state": "ACTIVE",
    "stateExplanation": "available",
    "stationnumber": 1126,
    "type": "ELEVATOR"
  }
]
)__"sv;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
U4,S1,U4,,
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
S3,01:15:00,25:15:00,3600
ICE,00:35:00,24:35:00,3600
U4,01:05:00,25:01:00,3600
)"sv;

void print_short(std::ostream& out, api::Itinerary const& j) {
  auto const format_time = [&](auto&& t, char const* fmt = "%F %H:%M") {
    out << date::format(fmt, *t);
  };
  auto const format_duration = [&](auto&& t, char const* fmt = "%H:%M") {
    out << date::format(fmt, std::chrono::milliseconds{t});
  };

  out << "date=";
  format_time(j.startTime_, "%F");
  out << ", start=";
  format_time(j.startTime_, "%H:%M");
  out << ", end=";
  format_time(j.endTime_, "%H:%M");

  out << ", duration=";
  format_duration(j.duration_ * 1000U);
  out << ", transfers=" << j.transfers_;

  out << ", legs=[\n";
  auto first = true;
  for (auto const& leg : j.legs_) {
    if (!first) {
      out << ",\n    ";
    } else {
      out << "    ";
    }
    first = false;
    out << "(";
    out << "from=" << leg.from_.stopId_.value_or("-")
        << " [track=" << leg.from_.track_.value_or("-")
        << ", scheduled_track=" << leg.from_.scheduledTrack_.value_or("-")
        << ", level=" << leg.from_.level_ << "]"
        << ", to=" << leg.to_.stopId_.value_or("-")
        << " [track=" << leg.to_.track_.value_or("-")
        << ", scheduled_track=" << leg.to_.scheduledTrack_.value_or("-")
        << ", level=" << leg.to_.level_ << "], ";
    out << "start=";
    format_time(leg.startTime_);
    out << ", mode=";
    out << json::serialize(json::value_from(leg.mode_));
    out << ", trip=\"" << leg.routeShortName_.value_or("-") << "\"";
    out << ", end=";
    format_time(leg.endTime_);
    out << ")";
  }
  out << "\n]";
}

struct trip_update {
  struct stop_time_update {
    std::string stop_id_;
    std::optional<std::uint32_t> seq_{std::nullopt};
    ::nigiri::event_type ev_type_{::nigiri::event_type::kDep};
    std::int32_t delay_minutes_{0U};
    bool skip_{false};
    std::optional<std::string> stop_assignment_{std::nullopt};
  };
  std::string trip_id_;
  std::optional<std::string> start_time_;
  std::optional<std::string> date_;
  std::vector<stop_time_update> stop_updates_{};
  bool cancelled_{false};
};

template <typename T>
std::uint64_t to_unix(T&& x) {
  return static_cast<std::uint64_t>(
      std::chrono::time_point_cast<std::chrono::seconds>(x)
          .time_since_epoch()
          .count());
};

transit_realtime::FeedMessage to_feed_msg(
    std::vector<trip_update> const& trip_delays,
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
    if (trip.cancelled_) {
      td->set_schedule_relationship(
          transit_realtime::TripDescriptor_ScheduleRelationship_CANCELED);
      continue;
    }
    if (trip.date_) {
      td->set_start_date(*trip.date_);
    }
    if (trip.start_time_) {
      td->set_start_time(*trip.start_time_);
    }

    for (auto const& stop_upd : trip.stop_updates_) {
      auto* const upd = e->mutable_trip_update()->add_stop_time_update();
      if (!stop_upd.stop_id_.empty()) {
        *upd->mutable_stop_id() = stop_upd.stop_id_;
      }
      if (stop_upd.seq_.has_value()) {
        upd->set_stop_sequence(*stop_upd.seq_);
      }
      if (stop_upd.stop_assignment_.has_value()) {
        upd->mutable_stop_time_properties()->set_assigned_stop_id(
            stop_upd.stop_assignment_.value());
      }
      if (stop_upd.skip_) {
        upd->set_schedule_relationship(
            transit_realtime::
                TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED);
        continue;
      }
      stop_upd.ev_type_ == ::nigiri::event_type::kDep
          ? upd->mutable_departure()->set_delay(stop_upd.delay_minutes_ * 60)
          : upd->mutable_arrival()->set_delay(stop_upd.delay_minutes_ * 60);
    }
  }

  return msg;
}

TEST(motis, routing) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .tiles_ = {{.profile_ = "deps/tiles/profile/full.lua",
                  .db_size_ = 1024U * 1024U * 25U}},
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}},
      .gbfs_ = {{.feeds_ = {{"CAB", {.url_ = "./test/resources/gbfs"}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true,
      .geocoding_ = true,
      .reverse_geocoding_ = true};
  auto d = import(c, "test/data", true);
  d.rt_->e_ = std::make_unique<elevators>(*d.w_, *d.elevator_nodes_,
                                          parse_fasta(kFastaJson));
  d.init_rtt(date::sys_days{2019_y / May / 1});

  {
    auto ioc = boost::asio::io_context{};
    boost::asio::co_spawn(
        ioc,
        [&]() -> boost::asio::awaitable<void> {
          co_await gbfs::update(c, *d.w_, *d.l_, d.gbfs_);
        },
        boost::asio::detached);
    ioc.run();
  }

  auto const stats = n::rt::gtfsrt_update_msg(
      *d.tt_, *d.rt_->rtt_, n::source_idx_t{0}, "test",
      to_feed_msg(
          {trip_update{.trip_id_ = "ICE",
                       .start_time_ = {"03:35:00"},
                       .date_ = {"20190501"},
                       .stop_updates_ = {{.stop_id_ = "FFM_12",
                                          .seq_ = std::optional{1U},
                                          .ev_type_ = n::event_type::kArr,
                                          .delay_minutes_ = 10,
                                          .stop_assignment_ = "FFM_12"}}}},
          date::sys_days{2019_y / May / 1} + 9h));
  EXPECT_EQ(1U, stats.total_entities_success_);

  auto const routing = utl::init_from<ep::routing>(d).value();
  EXPECT_EQ(d.rt_->rtt_.get(), routing.rt_->rtt_.get());

  // Route direct with GBFS.
  {
    auto const plan_response = routing(
        "?fromPlace=49.87526849014631,8.62771903392948"
        "&toPlace=49.87253873915287,8.629724234688751"
        "&time=2019-05-01T01:25Z"
        "&timetableView=false"
        "&directModes=WALK,RENTAL");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.direct_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:25, end=01:38, duration=00:13, transfers=0, legs=[
    (from=- [track=-, scheduled_track=-, level=0], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 01:25, mode="WALK", trip="-", end=2019-05-01 01:28),
    (from=- [track=-, scheduled_track=-, level=0], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 01:28, mode="RENTAL", trip="-", end=2019-05-01 01:29),
    (from=- [track=-, scheduled_track=-, level=0], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 01:29, mode="WALK", trip="-", end=2019-05-01 01:38)
])",
        ss.str());
  }

  // Route with GBFS.
  {
    auto const plan_response = routing(
        "?fromPlace=49.875308,8.6276673"
        "&toPlace=50.11347,8.67664"
        "&time=2019-05-01T01:25Z"
        "&timetableView=false"
        "&useRoutedTransfers=true"
        "&preTransitModes=WALK,RENTAL");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.itineraries_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:25, end=03:14, duration=01:49, transfers=1, legs=[
    (from=- [track=-, scheduled_track=-, level=0], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 01:25, mode="WALK", trip="-", end=2019-05-01 01:28),
    (from=- [track=-, scheduled_track=-, level=0], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 01:28, mode="RENTAL", trip="-", end=2019-05-01 01:29),
    (from=- [track=-, scheduled_track=-, level=0], to=test_DA_10 [track=10, scheduled_track=10, level=-1], start=2019-05-01 01:29, mode="WALK", trip="-", end=2019-05-01 01:37),
    (from=test_DA_10 [track=10, scheduled_track=10, level=-1], to=test_FFM_10 [track=10, scheduled_track=10, level=0], start=2019-05-01 02:35, mode="HIGHSPEED_RAIL", trip="ICE", end=2019-05-01 02:45),
    (from=test_FFM_10 [track=10, scheduled_track=10, level=0], to=test_de:6412:10:6:1 [track=U4, scheduled_track=U4, level=-2], start=2019-05-01 02:45, mode="WALK", trip="-", end=2019-05-01 02:49),
    (from=test_de:6412:10:6:1 [track=U4, scheduled_track=U4, level=-2], to=test_FFM_HAUPT_U [track=-, scheduled_track=-, level=-4], start=2019-05-01 03:05, mode="SUBWAY", trip="U4", end=2019-05-01 03:10),
    (from=test_FFM_HAUPT_U [track=-, scheduled_track=-, level=-4], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 03:10, mode="WALK", trip="-", end=2019-05-01 03:14)
])",
        ss.str());
  }

  // Route with wheelchair.
  {
    auto const plan_response = routing(
        "?fromPlace=49.87263,8.63127"
        "&toPlace=50.11347,8.67664"
        "&time=2019-05-01T01:25Z"
        "&pedestrianProfile=WHEELCHAIR"
        "&useRoutedTransfers=true"
        "&timetableView=false");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.itineraries_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:29, end=02:28, duration=01:03, transfers=1, legs=[
    (from=- [track=-, scheduled_track=-, level=0], to=test_DA_10 [track=10, scheduled_track=10, level=-1], start=2019-05-01 01:29, mode="WALK", trip="-", end=2019-05-01 01:35),
    (from=test_DA_10 [track=10, scheduled_track=10, level=-1], to=test_FFM_12 [track=12, scheduled_track=10, level=0], start=2019-05-01 01:35, mode="HIGHSPEED_RAIL", trip="ICE", end=2019-05-01 01:55),
    (from=test_FFM_12 [track=12, scheduled_track=10, level=0], to=test_FFM_101 [track=101, scheduled_track=101, level=-3], start=2019-05-01 01:55, mode="WALK", trip="-", end=2019-05-01 02:01),
    (from=test_FFM_101 [track=101, scheduled_track=101, level=-3], to=test_FFM_HAUPT_S [track=-, scheduled_track=-, level=-3], start=2019-05-01 02:15, mode="METRO", trip="S3", end=2019-05-01 02:20),
    (from=test_FFM_HAUPT_S [track=-, scheduled_track=-, level=-3], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 02:20, mode="WALK", trip="-", end=2019-05-01 02:28)
])",
        ss.str());
  }

  // Route without wheelchair.
  {
    auto const plan_response = routing(
        "?fromPlace=49.87263,8.63127"
        "&toPlace=50.11347,8.67664"
        "&time=2019-05-01T01:25Z"
        "&useRoutedTransfers=true"
        "&timetableView=false");

    auto ss = std::stringstream{};
    for (auto const& j : plan_response.itineraries_) {
      print_short(ss, j);
    }

    EXPECT_EQ(
        R"(date=2019-05-01, start=01:25, end=02:14, duration=00:49, transfers=1, legs=[
    (from=- [track=-, scheduled_track=-, level=0], to=test_DA_10 [track=10, scheduled_track=10, level=-1], start=2019-05-01 01:25, mode="WALK", trip="-", end=2019-05-01 01:28),
    (from=test_DA_10 [track=10, scheduled_track=10, level=-1], to=test_FFM_12 [track=12, scheduled_track=10, level=0], start=2019-05-01 01:35, mode="HIGHSPEED_RAIL", trip="ICE", end=2019-05-01 01:55),
    (from=test_FFM_12 [track=12, scheduled_track=10, level=0], to=test_de:6412:10:6:1 [track=U4, scheduled_track=U4, level=-2], start=2019-05-01 01:55, mode="WALK", trip="-", end=2019-05-01 01:59),
    (from=test_de:6412:10:6:1 [track=U4, scheduled_track=U4, level=-2], to=test_FFM_HAUPT_U [track=-, scheduled_track=-, level=-4], start=2019-05-01 02:05, mode="SUBWAY", trip="U4", end=2019-05-01 02:10),
    (from=test_FFM_HAUPT_U [track=-, scheduled_track=-, level=-4], to=- [track=-, scheduled_track=-, level=0], start=2019-05-01 02:10, mode="WALK", trip="-", end=2019-05-01 02:14)
])",
        ss.str());
  }
}
