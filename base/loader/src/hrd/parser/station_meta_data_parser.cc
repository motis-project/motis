#include "motis/loader/hrd/parser/station_meta_data_parser.h"

#include <cinttypes>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "boost/algorithm/string/trim.hpp"

#include "utl/parser/cstr.h"
#include "utl/parser/csv.h"
#include "utl/verify.h"

namespace motis::loader::hrd {

using namespace utl;

constexpr int DEFAULT_CHANGE_TIME_LONG_DISTANCE = 5;
constexpr int DEFAULT_CHANGE_TIME_LOCAL_TRANSPORT = 2;

//   0: <Gültig-Ab-Datum>
//   1: <Gelöscht-Flag>
//   2: <Name>
// * 3: <EVA-Nummer>
// * 4: <[RL100-Code 1, ...,RL100-Code N]>
//   5: <IBNR>
//   6: <UIC-Nummer>
//   7: <Staat>
//   8: <Amtlicher-Gemeinde-schlüssel>
//   9: <Langname>
//   10: <KFZ-Kennzeichen>
//   11: <Kurzname_1>
//   12: ...
//   13: <Kurzname_n>
void parse_ds100_mappings(loaded_file const& infotext_file,
                          std::map<cstr, int>& ds100_to_eva_number) {
  enum { eva_number, ds100_code };
  using entry = std::tuple<int, cstr>;

  std::array<uint8_t, MAX_COLUMNS> column_map{};
  std::fill(begin(column_map), end(column_map), NO_COLUMN_IDX);
  column_map[3] = 0;
  column_map[4] = 1;

  std::vector<entry> entries;
  auto next = infotext_file.content();
  while (next) {
    next = skip_lines(next, [](cstr const& line) { return line.len < 38; });
    if (!next) {
      break;
    }
    auto row = read_row<entry, ':'>(next, column_map, 5);
    entry e;
    read(e, row);
    auto const eva_num = std::get<eva_number>(e);
    for_each_token(std::get<ds100_code>(e), ',',
                   [&ds100_to_eva_number, &eva_num](cstr token) {
                     ds100_to_eva_number[token] = eva_num;
                   });
  }
}

enum { from_ds100_key, to_ds100_key, duration_key, track_change_time_key };
using minct = std::tuple<cstr, cstr, int, int>;
std::vector<minct> load_minct(loaded_file const& minct_file) {
  std::array<column_idx_t, MAX_COLUMNS> column_map{};
  std::fill(begin(column_map), end(column_map), NO_COLUMN_IDX);
  column_map[0] = 0;
  column_map[1] = 1;
  column_map[2] = 2;
  column_map[3] = 3;
  cstr minct_content(minct_file.content());
  auto rows = read_rows<minct, ';'>(minct_content, column_map);
  std::vector<minct> records;
  read(records, rows);
  return records;
}

void load_platforms(loaded_file const& platform_file,
                    station_meta_data& metas) {
  enum { ds100_code, station_name, platform_name, track_name };
  using entry = std::tuple<cstr, cstr, cstr, cstr>;

  std::vector<entry> entries;
  auto platform_content = platform_file.content();
  read<entry, ';'>(platform_content, entries,
                   {{"ril100", "bahnhof", "Bstg", "Gleis1"}});

  for (auto const& e : entries) {
    auto ds100 = std::get<ds100_code>(e).to_str();
    boost::algorithm::trim(ds100);
    metas.platforms_[ds100][std::get<platform_name>(e).to_str()].insert(
        std::get<track_name>(e).to_str());
  }
}

std::pair<int, int> station_meta_data::get_station_change_time(
    int eva_num) const {
  auto it = station_change_times_.find(eva_num);
  if (it == std::end(station_change_times_)) {
    if (eva_num < 1000000) {
      return {DEFAULT_CHANGE_TIME_LOCAL_TRANSPORT, 0};
    } else {
      return {DEFAULT_CHANGE_TIME_LONG_DISTANCE, 0};
    }
  } else {
    return it->second;
  }
}

void parse_and_add(loaded_file const& metabhf_file,
                   std::set<station_meta_data::footpath>& footpaths,
                   std::set<station_meta_data::meta_station>& meta_stations,
                   config const& c) {
  for_each_line(metabhf_file.content(), [&](cstr line) {
    if (line.length() < 16 || line[0] == '%' || line[0] == '*') {
      return;
    }

    if (line[7] == ':') {  // equivalent stations
      auto const eva = parse<int>(line.substr(c.meta_.meta_stations_.eva_));
      std::vector<int> equivalent;
      for_each_token(line.substr(8), ' ', [&c, &equivalent](cstr token) {
        if (c.version_ == "hrd_5_20_26" && token.starts_with("F")) {
          return;
        }
        auto const e = parse<int>(token);
        if (e != 0) {
          equivalent.push_back(e);
        }
      });

      if (!equivalent.empty()) {
        meta_stations.insert({eva, equivalent});
      }
    } else {  // footpaths
      auto f_equal = false;
      if (c.version_ == "hrd_5_00_8") {
        f_equal = line.length() > 23 ? line.substr(23, size(1)) == "F" : false;
      };

      footpaths.insert({parse<int>(line.substr(c.meta_.footpaths_.from_)),
                        parse<int>(line.substr(c.meta_.footpaths_.to_)),
                        parse<int>(line.substr(c.meta_.footpaths_.duration_)),
                        f_equal});
    }
  });
}

void parse_station_meta_data(loaded_file const& infotext_file,
                             loaded_file const& metabhf_file,
                             loaded_file const& metabhf_zusatz_file,
                             loaded_file const& minct_file,
                             loaded_file const& platform_file,
                             station_meta_data& metas, config const& config) {
  parse_ds100_mappings(infotext_file, metas.ds100_to_eva_num_);
  load_platforms(platform_file, metas);
  for (auto const& record : load_minct(minct_file)) {
    auto const from_ds100 = std::get<from_ds100_key>(record);
    auto const to_ds100 = std::get<to_ds100_key>(record);
    auto const duration = std::get<duration_key>(record);
    auto const platform_interchange = std::get<track_change_time_key>(record);

    if (to_ds100.len == 0) {
      auto eva_number_it = metas.ds100_to_eva_num_.find(from_ds100);
      if (eva_number_it != end(metas.ds100_to_eva_num_)) {
        metas.station_change_times_[eva_number_it->second] = {
            duration, platform_interchange};
      }
    } else {
      auto from_eva_num_it = metas.ds100_to_eva_num_.find(from_ds100);
      auto to_eva_num_it = metas.ds100_to_eva_num_.find(to_ds100);
      if (from_eva_num_it != end(metas.ds100_to_eva_num_) &&
          to_eva_num_it != end(metas.ds100_to_eva_num_)) {
        metas.footpaths_.insert(
            {from_eva_num_it->second, to_eva_num_it->second, duration, false});
      }
    }
  }
  parse_and_add(metabhf_file, metas.footpaths_, metas.meta_stations_, config);
  parse_and_add(metabhf_zusatz_file, metas.footpaths_, metas.meta_stations_,
                config);
}

}  // namespace motis::loader::hrd
