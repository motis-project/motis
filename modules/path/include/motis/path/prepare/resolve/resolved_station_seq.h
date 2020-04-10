#pragma once

#include <string>
#include <vector>

#include "motis/path/prepare/osm_path.h"

namespace motis::path {

struct sequence_info {
  sequence_info(size_t const idx, size_t const from, size_t const to,
                std::string type)
      : idx_(idx), from_(from), to_(to), type_(std::move(type)) {}

  size_t idx_;
  size_t from_;
  size_t to_;
  std::string type_;
};

struct resolved_station_seq {
  resolved_station_seq() = default;
  resolved_station_seq(std::vector<std::string> station_ids,
                       std::vector<uint32_t> classes,
                       std::vector<osm_path> paths,
                       std::vector<sequence_info> sequence_infos)
      : station_ids_{std::move(station_ids)},
        classes_{std::move(classes)},
        paths_{std::move(paths)},
        sequence_infos_{std::move(sequence_infos)} {}

  std::vector<std::string> station_ids_;
  std::vector<uint32_t> classes_;
  std::vector<osm_path> paths_;
  std::vector<sequence_info> sequence_infos_;
};

void write_to_fbs(std::vector<resolved_station_seq> const&,
                  std::string const& fname);
std::vector<resolved_station_seq> read_from_fbs(std::string const& fname);

}  // namespace motis::path
