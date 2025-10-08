#include "gtest/gtest.h"

#include "boost/url/url.hpp"

#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/import.h"
#include "motis/tag_lookup.h"

using namespace std::string_view_literals;
using namespace osr;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA 10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
+FFM_HÄUPT_&U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3 ,DB,S3,,,109
Ü4,DB,U4,,,402
+ICE_&A,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3 ,S1,S3 ,,
Ü4,S1,Ü4,,
+ICE_&A,S1,+ICE_&A,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3 ,01:15:00,01:15:00,FFM_101,1,0,0
S3 ,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
Ü4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
Ü4,01:10:00,01:10:00,+FFM_HÄUPT_&U,1,0,0
+ICE_&A,00:35:00,00:35:00,DA 10,0,0,0
+ICE_&A,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)"sv;

TEST(motis, tag_lookup) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = motis::config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ = motis::config::timetable{
          .first_day_ = "2019-05-01",
          .num_days_ = 2,
          .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}}};
  auto d = motis::import(c, "test/data", true);
  auto const rt = d.rt_;
  auto const rtt = rt->rtt_.get();
  EXPECT_TRUE(
      d.tags_->get_trip(*d.tt_, rtt, "20190501_01:15_test_S3 ").first.valid());
  EXPECT_TRUE(
      d.tags_->get_trip(*d.tt_, rtt, "20190501_01:05_test_Ü4").first.valid());
  EXPECT_TRUE(d.tags_->get_trip(*d.tt_, rtt, "20190501_00:35_test_+ICE_&A")
                  .first.valid());
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_DA 10"));
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_+FFM_HÄUPT_&U"));
  EXPECT_NE(nigiri::location_idx_t::invalid(),
            d.tags_->get_location(*d.tt_, "test_DA 10"));
  auto u = boost::urls::url{
      "/api",
  };
  u.params({true, false, false}).append({"encoded", "a+& b"});
  u.params({false, false, false}).append({"unencoded", "a+& b"});
  {
    std::stringstream buffer;
    buffer << u;
    EXPECT_EQ("/api?encoded=a%2B%26+b&unencoded=a+%26%20b", buffer.str());
  }
  {
    std::stringstream buffer;
    buffer << u.encoded_params();
    EXPECT_EQ("encoded=a%2B%26+b&unencoded=a+%26%20b", buffer.str());
  }
}
