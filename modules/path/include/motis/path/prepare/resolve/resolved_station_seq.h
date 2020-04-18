#pragma once

#include <string>
#include <vector>

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

struct sequence_info {
  sequence_info(size_t idx, size_t from, size_t to, source_spec spec)
      : idx_(idx), from_(from), to_(to), source_spec_(spec) {}

  size_t idx_, from_, to_;
  source_spec source_spec_;
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
