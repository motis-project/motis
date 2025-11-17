#include "gtest/gtest.h"

#include "osr/platforms.h"
#include "osr/routing/profile.h"
#include "osr/routing/profiles/bike.h"
#include "osr/routing/profiles/bike_sharing.h"
#include "osr/routing/profiles/car.h"
#include "osr/routing/profiles/car_sharing.h"
#include "osr/routing/profiles/foot.h"
#include "osr/types.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "motis/config.h"
#include "motis/import.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"

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

TEST(motis, get_track) {
  ASSERT_FALSE(motis::get_track("a:").has_value());

  auto const track = motis::get_track("a:232");
  ASSERT_TRUE(track.has_value());
  EXPECT_EQ("232", *track);

  auto const track_1 = motis::get_track("232");
  ASSERT_TRUE(track_1.has_value());
  EXPECT_EQ("232", *track_1);
}

TEST(motis, get_way_candidates) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = motis::config{
      .server_ = {{.web_folder_ = "ui/build", .n_threads_ = 1U}},
      .osm_ = {"test/resources/test_case.osm.pbf"},
      .timetable_ =
          motis::config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .use_osm_stop_coordinates_ = true,
              .extend_missing_footpaths_ = false,
              .preprocess_max_matching_distance_ = 250,
              .datasets_ = {{"test", {.path_ = std::string{kGTFS}}}}},
      .street_routing_ = true,
      .osr_footpath_ = true,
      .geocoding_ = true,
      .reverse_geocoding_ = true};
  auto d = motis::import(c, "test/data", true);
  auto const location_idxs =
      utl::to_vec(utl::enumerate(d.tt_->locations_.src_),
                  [&](std::tuple<size_t, nigiri::source_idx_t> const ll) {
                    return nigiri::location_idx_t{std::get<0>(ll)};
                  });
  auto const locs =
      utl::to_vec(location_idxs, [&](nigiri::location_idx_t const l) {
        return osr::location{
            d.tt_->locations_.coordinates_[nigiri::location_idx_t{l}],
            d.pl_->get_level(*d.w_, (*d.matches_)[nigiri::location_idx_t{l}])};
      });

  auto const get_path = [&](search_profile const p, way_candidate const& a,
                            node_candidate const& anc, location const& l) {
    switch (p) {
      case search_profile::kFoot: [[fallthrough]];
      case search_profile::kWheelchair: [[fallthrough]];
      case search_profile::kCar: [[fallthrough]];
      case search_profile::kBike: [[fallthrough]];
      case search_profile::kCarSharing: [[fallthrough]];
      case search_profile::kBikeSharing:
        return d.l_->get_node_candidate_path(a, anc, true, l);
      default: return std::vector<geo::latlng>{};
    }
  };

  for (auto profile :
       {osr::search_profile::kCar, osr::search_profile::kCarSharing,
        osr::search_profile::kFoot, osr::search_profile::kWheelchair,
        osr::search_profile::kBike, osr::search_profile::kBikeSharing}) {
    auto const with_preprocessing = motis::get_reverse_platform_way_matches(
        *d.l_, &*d.way_matches_, profile, location_idxs, locs,
        osr::direction::kForward, 250);
    auto const without_preprocessing = utl::to_vec(
        utl::zip(location_idxs, locs),
        [&](std::tuple<nigiri::location_idx_t, osr::location> const ll) {
          auto const& [l, query] = ll;
          return d.l_->match(motis::to_profile_parameters(profile, {}), query,
                             true, osr::direction::kForward, 250, nullptr,
                             profile);
        });

    ASSERT_EQ(with_preprocessing.size(), without_preprocessing.size());
    for (auto const [with, without, l] :
         utl::zip(with_preprocessing, without_preprocessing, locs)) {
      ASSERT_EQ(with.size(), without.size());
      auto sorted_with = with;
      auto sorted_without = without;
      auto const sort_by_way = [&](auto const& a, auto const& b) {
        return a.way_ < b.way_;
      };
      utl::sort(sorted_with, sort_by_way);
      utl::sort(sorted_without, sort_by_way);
      for (auto [a, b] : utl::zip(sorted_with, sorted_without)) {
        ASSERT_FLOAT_EQ(a.dist_to_way_, b.dist_to_way_);
        ASSERT_EQ(a.way_, b.way_);
        for (auto const& [anc, bnc] :
             {std::tuple{a.left_, b.left_}, std::tuple{a.right_, b.right_}}) {
          ASSERT_EQ(anc.node_, bnc.node_);
          if (anc.valid()) {
            EXPECT_FLOAT_EQ(anc.dist_to_node_, bnc.dist_to_node_);
            EXPECT_EQ(anc.cost_, bnc.cost_);
            EXPECT_EQ(anc.lvl_, bnc.lvl_);
            EXPECT_EQ(anc.way_dir_, bnc.way_dir_);
            EXPECT_EQ(get_path(profile, a, anc, l), bnc.path_);
          }
        }
      }
    }

    auto const with_preprocessing_but_larger =
        motis::get_reverse_platform_way_matches(*d.l_, &*d.way_matches_,
                                                profile, location_idxs, locs,
                                                osr::direction::kForward, 500);

    auto const with_preprocessing_but_smaller =
        motis::get_reverse_platform_way_matches(*d.l_, &*d.way_matches_,
                                                profile, location_idxs, locs,
                                                osr::direction::kForward, 50);

    for (auto const [with, larger, smaller] :
         utl::zip(with_preprocessing, with_preprocessing_but_larger,
                  with_preprocessing_but_smaller)) {
      if (with.size() == 0 && larger.size() == 0 && smaller.size() == 0) {
        continue;
      }
      EXPECT_GE(larger.size(), with.size());
      EXPECT_GE(with.size(), smaller.size());
      auto const& a = larger[0];
      auto const& b = smaller[0];
      EXPECT_TRUE(!a.left_.valid() ||
                  a.left_.path_.size() != 0);  // on the fly match
      EXPECT_TRUE(!b.left_.valid() ||
                  b.left_.path_.size() == 0);  // preprocessed match
    }

    for (auto dist : {5, 10, 25, 250, 1000}) {
      auto const remote_station =
          osr::location{{49.8731904, 8.6221451}, level_t{}};
      auto const raw = d.l_->get_raw_match(remote_station, dist);
      auto const params = motis::to_profile_parameters(profile, {});
      auto const with =
          d.l_->match(params, remote_station, true, osr::direction::kForward,
                      dist, nullptr, profile, raw);
      auto const without =
          d.l_->match(params, remote_station, true, osr::direction::kForward,
                      dist, nullptr, profile);
      EXPECT_NE(0, raw.size());
      EXPECT_EQ(with.size(), without.size());
    }
  }
}
