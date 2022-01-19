#include <cstring>

#include "gtest/gtest.h"

#include "motis/loader/hrd/parser/station_meta_data_parser.h"
#include "motis/loader/hrd/parser/stations_parser.h"
#include "motis/loader/parser_error.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

constexpr char const* infotext = R"(%
0000001 ICE International
0000002 Canopus
0000003 Metropol
0000004 Kopernikus
0088506 2002-05-01:N:Darmstadt Hbf:8000068:FD:68:008011336:DE:06411000:Darmstadt Hbf:DA
0599553 140852529:2015-08-18T00:01:00Z
0088540 2002-05-01:N:Frankfurt(Main)Hbf:8000105:FF:105:008011068:DE:06412000:Frankfurt(Main)Hbf
0599554 140852530:2015-08-18T00:01:00Z
0599555 140852531:2015-08-18T00:01:00Z
0599556 140852532:2015-08-18T00:01:00Z
)";

constexpr char const* footpaths_old = R"(%
8003919 0721747 015
8003935 0651301 003
8089221: 0122662
8089222: 8006552 8089222
)";

constexpr char const* footpaths_new = R"(%
8003918 0721747 015
*A XQ
*U 1
8003935 0651301 005

0663370: F0651591  0663370
)";

constexpr char const* footpaths_hrd_20_26 = R"(%
8003919 0721747 015
8003935 0651301 003
8089221:  0122662
8089222:  8006552  8089222
)";

constexpr char const* stations_data = R"(%
0100001     Hauptwache, Frankfurt am Main
0100002     Roemer/Paulskirche, Frankfurt am Main
)";

constexpr char const* coordinates_data = R"(%
0100001   8.679296  50.113963 Hauptwache, Frankfurt am Main
0100002   8.681793  50.110902 Roemer/Paulskirche, Frankfurt am Main
)";

constexpr auto const minct_data = R"(FD;;7;3
FF;FFT;10;-
FF;;8;4)";

constexpr char const* platform_data = R"(ril100;bahnhof;Bstg;Gleis1;Kennzeichen
FD ;Darmstadt Hbf;1;1;p
FD ;Darmstadt Hbf;3-4;3;p
FD ;Darmstadt Hbf;3-4;4;p
FD ;Darmstadt Hbf;5-6;5;p
FD ;Darmstadt Hbf;5-6;6;p
FD ;Darmstadt Hbf;7-8;7;p
FD ;Darmstadt Hbf;7-8;8;p
FD ;Darmstadt Hbf;9-10;9;p
FD ;Darmstadt Hbf;9-10;10;p
FD ;Darmstadt Hbf;11-12;11;p
FD ;Darmstadt Hbf;11-12;12;p
FF ;Frankfurt (Main) Hbf;1/1A;1;p
FF ;Frankfurt (Main) Hbf;2/3;2;p
FF ;Frankfurt (Main) Hbf;2/3;3;p
FF ;Frankfurt (Main) Hbf;4/5;4;p
FF ;Frankfurt (Main) Hbf;4/5;5;p
FF ;Frankfurt (Main) Hbf;6/7;6;p
FF ;Frankfurt (Main) Hbf;6/7;7;p
FF ;Frankfurt (Main) Hbf;8/9;8;p
FF ;Frankfurt (Main) Hbf;8/9;9;p
FF ;Frankfurt (Main) Hbf;10/11;10;p
FF ;Frankfurt (Main) Hbf;10/11;11;p
FF ;Frankfurt (Main) Hbf;12/13;12;p
FF ;Frankfurt (Main) Hbf;12/13;13;p
FF ;Frankfurt (Main) Hbf;14/15;14;p
FF ;Frankfurt (Main) Hbf;14/15;15;p
FF ;Frankfurt (Main) Hbf;16/17;16;p
FF ;Frankfurt (Main) Hbf;16/17;17;p
FF ;Frankfurt (Main) Hbf;18/19;18;p
FF ;Frankfurt (Main) Hbf;18/19;19;p
FF ;Frankfurt (Main) Hbf;20/21;20;p
FF ;Frankfurt (Main) Hbf;20/21;21;p
FF ;Frankfurt (Main) Hbf;22/23;22;p
FF ;Frankfurt (Main) Hbf;22/23;23;p
FF ;Frankfurt (Main) Hbf;24;24;p
FF ;Frankfurt (Main) Hbf;S101/102;101;p
FF ;Frankfurt (Main) Hbf;S101/102;102;p
FF ;Frankfurt (Main) Hbf;S103/104;103;p
FF ;Frankfurt (Main) Hbf;S103/104;104;p
FF ;Frankfurt (Main) Hbf;1/1A;1a;p
FF ;Frankfurt (Main) Hbf;U1;U1;p
FF ;Frankfurt (Main) Hbf;U2;U2;p
)";

TEST(loader_hrd_stations_parser, meta_data) {
  try {
    loaded_file info_text_file("infotext.101", infotext);
    loaded_file fp_old_file("footpaths_old.101", footpaths_old);
    loaded_file fp_new_file("footpaths_new.101", footpaths_new);
    loaded_file fp_new_file_2("footpats_new_2.101", footpaths_hrd_20_26);
    loaded_file minct_csv("minct.csv", minct_data);
    loaded_file platform_csv("platform.csv", platform_data);

    station_meta_data meta_old;
    parse_station_meta_data(info_text_file, fp_old_file, fp_new_file, minct_csv,
                            platform_csv, meta_old, hrd_5_00_8);
    station_meta_data meta_new;
    parse_station_meta_data(info_text_file, fp_new_file_2, fp_new_file,
                            minct_csv, platform_csv, meta_new, hrd_5_20_26);

    for (auto m : {meta_old, meta_new}) {
      ASSERT_EQ(2, m.station_change_times_.size());
      ASSERT_EQ(std::make_pair(7, 3), m.get_station_change_time(8000068));
      ASSERT_EQ(std::make_pair(8, 4), m.get_station_change_time(8000105));

      ASSERT_EQ(3, m.footpaths_.find({8003935, 651301, -1, false})->duration_);
    }
  } catch (parser_error const& pe) {
    pe.print_what();
    ASSERT_TRUE(false);
  }
}

TEST(loader_hrd_stations_parser, parse_stations) {
  station_meta_data meta_old;
  station_meta_data meta_new;

  auto const infotext_file = loaded_file{"infotext.101", infotext};
  auto const metabhf_file = loaded_file{"metabhf.101", footpaths_old};
  auto const meta_zusatz_file =
      loaded_file{"metabhf_zusatz.101", footpaths_new};
  auto const meta_hrd_20_26_file =
      loaded_file{"metabhf.101", footpaths_hrd_20_26};

  parse_station_meta_data(infotext_file, metabhf_file, meta_zusatz_file,
                          loaded_file{}, loaded_file{}, meta_old, hrd_5_00_8);
  parse_station_meta_data(infotext_file, meta_hrd_20_26_file, meta_zusatz_file,
                          loaded_file{}, loaded_file{}, meta_new, hrd_5_20_26);

  auto const bahnhof = loaded_file{"bahnhof.101", stations_data};
  auto const koords = loaded_file{"dbkoords.101", coordinates_data};
  auto stations_old = parse_stations(bahnhof, koords, meta_old, hrd_5_00_8);
  auto stations_new = parse_stations(bahnhof, koords, meta_new, hrd_5_20_26);

  for (auto stations : {stations_old, stations_new}) {
    ASSERT_EQ(2, stations.size());

    auto it = stations.find(100001);
    ASSERT_TRUE(it != end(stations));

    auto station = it->second;
    ASSERT_STREQ("Hauptwache, Frankfurt am Main", station.name_.c_str());
    ASSERT_TRUE(std::abs(station.lng_ - 8.679296) <= 0.001);
    ASSERT_TRUE(std::abs(station.lat_ - 50.113963) <= 0.001);
    ASSERT_EQ(2, station.change_time_);

    it = stations.find(100002);
    ASSERT_TRUE(it != end(stations));

    station = it->second;
    ASSERT_STREQ("Roemer/Paulskirche, Frankfurt am Main",
                 station.name_.c_str());
    ASSERT_TRUE(std::abs(station.lng_ - 8.681793) <= 0.001);
    ASSERT_TRUE(std::abs(station.lat_ - 50.110902) <= 0.001);
    ASSERT_EQ(2, station.change_time_);
  }

  for (auto const& metas : {meta_old, meta_new}) {
    auto const& meta_stations = metas.meta_stations_;
    ASSERT_EQ(3, meta_stations.size());
    for (auto const& meta_station : meta_stations) {
      if (meta_station.eva_ == 8089221) {
        ASSERT_EQ(1, meta_station.equivalent_.size());
        EXPECT_EQ(122662, meta_station.equivalent_[0]);
      }

      if (meta_station.eva_ == 8006552) {
        ASSERT_EQ(2, meta_station.equivalent_.size());
        EXPECT_EQ(8006552, meta_station.equivalent_[0]);
        EXPECT_EQ(8089222, meta_station.equivalent_[1]);
      }

      if (meta_station.eva_ == 0663370) {
        ASSERT_EQ(2, meta_station.equivalent_.size());
        EXPECT_EQ(651591, meta_station.equivalent_[0]);
        EXPECT_EQ(663370, meta_station.equivalent_[1]);
      }
    }
  }
}

}  // namespace motis::loader::hrd
