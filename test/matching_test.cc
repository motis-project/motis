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

#include "motis/data.h"
#include "motis/match_platforms.h"
#include "motis/osr/parameters.h"

#include "./test_case.h"

using namespace osr;

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
  auto [d, _] = get_test_case<test_case::FFM_get_way_candidates>();
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
