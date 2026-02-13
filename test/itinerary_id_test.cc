#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "motis/config.h"
#include "motis/endpoints/routing.h"
#include "motis/import.h"
#include "motis/itinerary_id.h"

using namespace motis;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
{},DA Hbf,49.87260,8.63085,1,,
{},FFM Hbf,50.10701,8.66341,1,,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
ICE,10:00:00,10:35:00,{},0,0,0
ICE,11:00:00,11:00:00,{},1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1
)";

TEST(motis, itinerary_id) {
  std::cout << "TEST(motis, itinerary_id) running..." << std::endl;
  auto c = config{
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test",
                             {.path_ = std::string{fmt::format(
                                  kGTFS, "DA", "FFM", "DA", "FFM")}}}}},
  };

  auto original = api::Itinerary{};
  {
    auto d = import(c, "test/data", true);
    auto const routing = utl::init_from<ep::routing>(d).value();
    original = routing(
                   "?fromPlace=test_DA"
                   "&toPlace=test_FFM"
                   "&time=2019-05-01T02:00Z"
                   "&timetableView=false"
                   "&directModes=WALK,RENTAL")
                   .itineraries_.at(0);
  }

  {
    c.timetable_->datasets_.at("test").path_ =
        std::string{fmt::format(kGTFS, "FFM", "DA", "FFM", "DA")};

    auto d = import(c, "test/data", true);
    auto const routing = utl::init_from<ep::routing>(d).value();
    auto const expected = routing(
                              "?fromPlace=test_FFM"
                              "&toPlace=test_DA"
                              "&time=2019-05-01T02:00Z"
                              "&timetableView=false"
                              "&directModes=WALK,RENTAL")
                              .itineraries_.at(0);

    auto const id_ = generate_itinerary_id(original);
    std::cout << "id_: " << id_ << '\n' << std::endl;

    EXPECT_EQ(expected,
              reconstruct_itinerary(
                  motis::ep::stops{.config_ = d.config_,
                                   .w_ = d.w_.get(),
                                   .pl_ = d.pl_.get(),
                                   .matches_ = d.matches_.get(),
                                   .ae_ = d.adr_ext_.get(),
                                   .tz_ = d.tz_.get(),
                                   .loc_rtree_ = *d.location_rtree_,
                                   .tags_ = *d.tags_,
                                   .tt_ = *d.tt_},
                  motis::ep::stop_times{.config_ = d.config_,
                                        .w_ = d.w_.get(),
                                        .pl_ = d.pl_.get(),
                                        .matches_ = d.matches_.get(),
                                        .ae_ = d.adr_ext_.get(),
                                        .tz_ = d.tz_.get(),
                                        .loc_rtree_ = *d.location_rtree_,
                                        .tt_ = *d.tt_,
                                        .tags_ = *d.tags_,
                                        .rt_ = d.rt_},
                  id_));
  }
}
