#pragma once

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "motis/loader/hrd/parse_config.h"
#include "motis/loader/loaded_file.h"

namespace motis::loader::hrd {

struct station_meta_data {
  struct footpath {
    bool operator<(footpath const& rh) const {
      return std::tie(from_eva_num_, to_eva_num_) <
             std::tie(rh.from_eva_num_, rh.to_eva_num_);
    }
    int from_eva_num_;
    int to_eva_num_;
    int duration_;
    bool f_equal_;
  };

  struct meta_station {
    bool operator<(meta_station const& rh) const { return eva_ < rh.eva_; }
    int eva_;
    std::vector<int> equivalent_;
  };

  std::pair<int, int> get_station_change_time(int eva_num) const;

  std::map<int, std::pair<int, int>> station_change_times_;
  std::set<footpath> footpaths_;
  std::set<meta_station> meta_stations_;
  std::map<utl::cstr, int> ds100_to_eva_num_;
  std::map<std::string, std::map<std::string, std::set<std::string>>>
      platforms_;
};

void parse_station_meta_data(loaded_file const& infotext_file,
                             loaded_file const& metabhf_file,
                             loaded_file const& metabhf_zusatz_file,
                             loaded_file const& minct_file,
                             loaded_file const& platform_file,
                             station_meta_data&, config const&);

}  // namespace motis::loader::hrd
