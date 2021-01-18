#include "motis/loader/hrd/builder/station_builder.h"

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/util.h"

namespace motis::loader::hrd {

using namespace flatbuffers64;

station_builder::station_builder(
    std::map<int, intermediate_station> hrd_stations, timezones tz)
    : hrd_stations_(std::move(hrd_stations)), timezones_(std::move(tz)){};

Offset<Station> station_builder::get_or_create_station(int eva_num,
                                                       FlatBufferBuilder& fbb) {
  return utl::get_or_create(fbs_stations_, eva_num, [&]() {
    auto it = hrd_stations_.find(eva_num);
    utl::verify(it != end(hrd_stations_), "missing station: {}", eva_num);
    auto tze = timezones_.find(eva_num);
    return CreateStation(
        fbb, to_fbs_string(fbb, pad_to_7_digits(eva_num)),
        to_fbs_string(fbb, it->second.name_, ENCODING), it->second.lat_,
        it->second.lng_, it->second.change_time_,
        fbb.CreateVector(utl::to_vec(
            begin(it->second.ds100_), end(it->second.ds100_),
            [&](std::string const& s) { return fbb.CreateString(s); })),
        utl::get_or_create(
            fbs_timezones_, tze,
            [&]() {
              if (tze->season_) {
                auto const& season = *(tze->season_);
                return CreateTimezone(
                    fbb, tze->general_gmt_offset_,
                    CreateSeason(fbb, season.gmt_offset_, season.first_day_idx_,
                                 season.last_day_idx_,
                                 season.season_begin_time_,
                                 season.season_end_time_));
              } else {
                return CreateTimezone(fbb, tze->general_gmt_offset_);
              }
            }),
        0, it->second.platform_change_time_,
        fbb.CreateVector(
            utl::to_vec(it->second.platforms_, [&](auto const& platform) {
              return CreatePlatform(
                  fbb, fbb.CreateString(platform.first),
                  fbb.CreateVector(
                      utl::to_vec(platform.second, [&](auto const& track) {
                        return fbb.CreateString(track);
                      })));
            })));
  });
}

}  // namespace motis::loader::hrd
