#include "gtest/gtest.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"

#include "utl/init_from.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
Parent1,Parent1,50.0,8.0,1,,
Child1A,Child1A,50.001,8.001,0,Parent1,1
Child1B,Child1B,50.002,8.002,0,Parent1,2
Parent2,Parent2,51.0,9.0,1,,
Child2,Child2,51.001,9.001,0,Parent2,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,R1,,109
R2,DB,R2,R2,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,Parent2 Express,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,stop_headsign
T1,10:00:00,10:00:00,Child1A,1,0,0,Origin
T1,10:10:00,10:10:00,Child1B,2,0,0,Midway
T1,11:00:00,11:00:00,Child2,3,0,0,Destination

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# translations.txt
table_name,field_name,language,translation,record_id,record_sub_id,field_value
routes,route_long_name,de,DE-R1,,,R1
routes,route_long_name,fr,FR-R1,,,R1
routes,route_long_name,en,EN-R1,,,R1
stops,stop_name,en,Child1A,Child1A,,
stops,stop_name,de,Kind 1A,Child1A,,
stops,stop_name,en,Child1B,,,Child1B
stops,stop_name,de,Kind 1B,,,Child1B
stops,stop_name,en,Parent2,Parent2,,
stops,stop_name,de,Eltern 2,Parent2,,
stops,stop_name,fr,Parent Deux,Parent2,,
stops,stop_name,fr,Enfant 1A,Child1A,,
stops,stop_name,fr,Enfant 1B,,,Child1B
stop_times,stop_headsign,en,Parent2 Express,T1,1,
stop_times,stop_headsign,de,Richtung Eltern Zwei,T1,1,
stop_times,stop_headsign,fr,Vers Parent Deux,T1,1,
)";

constexpr auto kScript = R"(
function process_route(route)
  route:set_short_name({
    translation.new('en', 'EN_SHORT_NAME'),
    translation.new('de', 'DE_SHORT_NAME'),
    translation.new('fr', 'FR_SHORT_NAME')
  })
  route:get_short_name_translations():add(translation.new('hu', 'HU_SHORT_NAME'))
  print(route:get_short_name_translations():get(1):get_text())
  print(route:get_short_name_translations():get(1):get_language())
end
)";

TEST(motis, trip_stop_naming) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = kGTFS, .script_ = kScript}}}},
      .street_routing_ = false};
  auto d = import(c, "test/data", true);

  auto const trip_ep = utl::init_from<ep::trip>(d).value();

  auto const res = trip_ep("?tripId=20190501_10%3A00_test_T1");
  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_[0];
  EXPECT_EQ("Child1A", leg.from_.name_);
  ASSERT_TRUE(leg.intermediateStops_.has_value());
  ASSERT_EQ(1, leg.intermediateStops_->size());
  EXPECT_EQ("Child1B", leg.intermediateStops_->at(0).name_);
  EXPECT_EQ("Parent2", leg.to_.name_);
  EXPECT_EQ("Parent2 Express", leg.headsign_);
  EXPECT_EQ("EN_SHORT_NAME", leg.routeShortName_);
  EXPECT_EQ("EN-R1", leg.routeLongName_);

  auto const res_de = trip_ep("?tripId=20190501_10%3A00_test_T1&language=de");
  ASSERT_EQ(1, res_de.legs_.size());
  auto const& leg_de = res_de.legs_[0];
  EXPECT_EQ("Kind 1A", leg_de.from_.name_);
  ASSERT_TRUE(leg_de.intermediateStops_.has_value());
  ASSERT_EQ(1, leg_de.intermediateStops_->size());
  EXPECT_EQ("Kind 1B", leg_de.intermediateStops_->at(0).name_);
  EXPECT_EQ("Eltern 2", leg_de.to_.name_);
  EXPECT_EQ("Richtung Eltern Zwei", leg_de.headsign_);
  EXPECT_EQ("DE_SHORT_NAME", leg_de.routeShortName_);
  EXPECT_EQ("DE-R1", leg_de.routeLongName_);

  auto const res_fr = trip_ep("?tripId=20190501_10%3A00_test_T1&language=fr");
  ASSERT_EQ(1, res_fr.legs_.size());
  auto const& leg_fr = res_fr.legs_[0];
  EXPECT_EQ("Enfant 1A", leg_fr.from_.name_);
  ASSERT_TRUE(leg_fr.intermediateStops_.has_value());
  ASSERT_EQ(1, leg_fr.intermediateStops_->size());
  EXPECT_EQ("Enfant 1B", leg_fr.intermediateStops_->at(0).name_);
  EXPECT_EQ("Parent Deux", leg_fr.to_.name_);
  EXPECT_EQ("Vers Parent Deux", leg_fr.headsign_);
  EXPECT_EQ("FR_SHORT_NAME", leg_fr.routeShortName_);
  EXPECT_EQ("FR-R1", leg_fr.routeLongName_);
}
