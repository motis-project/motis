#pragma once

#include "conf/configuration.h"

namespace motis::path {

struct prepare_settings : public conf::configuration {
  prepare_settings() : configuration("Prepare Options", "") {
    param(schedule_, "schedule", "/path/to/rohdaten");
    param(osm_, "osm", "/path/to/germany-latest.osm.pbf");
    param(osrm_, "osrm", "path/to/osrm/files");
    param(out_, "out", "/path/to/db.mdb");
    param(tmp_, "tmp", "/path/to/tmp/directory");
    param(filter_, "filter", "filter station sequences");
    param(osm_cache_task_, "osm_cache_task", "{ignore, load, dump}");
    param(osm_cache_file_, "osm_cache_file", "/path/to/osm_cache.bin");
    param(seq_cache_task_, "seq_cache_task", "{ignore, load, dump}");
    param(seq_cache_file_, "seq_cache_file", "/path/to/seq_cache.fbs");
  }

  std::string schedule_{"rohdaten"};
  std::string osm_{"germany-latest.osm.pbf"};
  std::string osrm_{"osrm"};
  std::string out_{"./pathdb.mdb"};
  std::string tmp_{"."};

  std::vector<std::string> filter_;

  std::string osm_cache_task_{"ignore"};
  std::string osm_cache_file_{"osm_cache.bin"};

  std::string seq_cache_task_{"ignore"};
  std::string seq_cache_file_{"seq_cache.bin"};
};

void prepare(prepare_settings const&);

}  // namespace motis::path
