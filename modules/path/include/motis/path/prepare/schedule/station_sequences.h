#pragma once

#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "cista/memory_holder.h"

#include "geo/latlng.h"

#include "motis/memory.h"
#include "motis/vector.h"

#include "motis/core/schedule/connection.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/source_spec.h"

namespace motis {

namespace loader {

struct Schedule;  // NOLINT

}  // namespace loader

namespace path {

struct sequence_info {
  size_t idx_, from_, to_;
  bool between_stations_;
  source_spec source_spec_;
};

struct station_seq {
  mcd::vector<mcd::string> station_ids_;
  mcd::vector<mcd::string> station_names_;
  mcd::vector<geo::latlng> coordinates_;

  mcd::vector<service_class> classes_;

  mcd::vector<osm_path> paths_;
  mcd::vector<sequence_info> sequence_infos_;
  float distance_{0};
};

mcd::vector<station_seq> load_station_sequences(motis::loader::Schedule const*);

mcd::unique_ptr<mcd::vector<station_seq>> read_station_sequences(
    std::string const& fname, cista::memory_holder&);
void write_station_sequences(std::string const& fname,
                             mcd::unique_ptr<mcd::vector<station_seq>> const&);

}  // namespace path
}  // namespace motis
