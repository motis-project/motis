#pragma once

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/filesystem.hpp"

#include "conf/configuration.h"

#include "geo/polygon.h"

#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/path/path_database.h"
#include "motis/path/prepare/db_builder.h"
#include "motis/path/prepare/post/build_post_graph.h"
#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/post/post_processor.h"
#include "motis/path/prepare/post/post_serializer.h"
#include "motis/path/prepare/resolve/path_routing.h"
#include "motis/path/prepare/resolve/resolve_sequences.h"
#include "motis/path/prepare/schedule/schedule_wrapper.h"
#include "motis/path/prepare/schedule/stations.h"
#include "motis/path/prepare/schedule/stop_positions.h"

namespace motis::path {

struct prepare_settings : public conf::configuration {
  prepare_settings() : configuration("Prepare Options", "") {
    param(schedule_, "schedule", "/path/to/rohdaten");
    param(osm_, "osm", "/path/to/germany-latest.osm.pbf");
    param(osrm_, "osrm", "path/to/osrm/files");
    param(out_, "out", "/path/to/db.mdb");
    param(filter_, "filter", "filter station sequences");
    param(seq_cache_task_, "seq_cache_task", "{ignore, load, dump}");
    param(seq_cache_file_, "seq_cache_file", "/path/to/seq_cache.fbs");
  }

  std::string schedule_{"rohdaten"};
  std::string osm_{"germany-latest.osm.pbf"};
  std::string osrm_{"osrm"};
  std::string out_{"./pathdb.mdb"};

  std::vector<std::string> filter_;

  std::string seq_cache_task_{"ignore"};
  std::string seq_cache_file_{"seq_cache.fbs"};
};

inline void filter_sequences(std::vector<std::string> const& filters,
                             std::vector<station_seq>& sequences) {
  logging::scoped_timer timer("filter station sequences");
  for (auto const& filter : filters) {
    std::vector<std::string> tokens;
    boost::split(tokens, filter, boost::is_any_of(":"));
    utl::verify(tokens.size() == 2, "unexpected filter");

    if (tokens[0] == "id") {
      utl::erase_if(sequences, [&tokens](auto const& seq) {
        return std::none_of(
            begin(seq.station_ids_), end(seq.station_ids_),
            [&tokens](auto const& id) { return id == tokens[1]; });
      });
    } else if (tokens[0] == "seq") {
      std::vector<std::string> ids;
      boost::split(ids, tokens[1], boost::is_any_of("."));
      utl::erase_if(sequences, [&ids](auto const& seq) {
        return ids != seq.station_ids_;
      });
    } else if (tokens[0] == "extent") {
      utl::verify(boost::filesystem::is_regular_file(tokens[1]),
                  "cannot find extent polygon");
      auto const extent_polygon = geo::read_poly_file(tokens[1]);
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::any_of(begin(seq.coordinates_), end(seq.coordinates_),
                           [&](auto const& coord) {
                             return !geo::within(coord, extent_polygon);
                           });
      });
    } else if (tokens[0] == "limit") {
      size_t const count = std::stoul(tokens[1]);
      sequences.resize(std::min(count, sequences.size()));
    } else if (tokens[0] == "cat") {
      auto cat = std::stoi(tokens[1]);
      utl::erase_if(sequences, [&](auto const& seq) {
        return seq.categories_.find(cat) == end(seq.categories_);
      });
      for (auto& seq : sequences) {
        seq.categories_ = {cat};
      }
    } else {
      LOG(logging::info) << "unknown filter: " << tokens[0];
    }
  }
}

inline void prepare(prepare_settings const& opt) {
  utl::verify(boost::filesystem::is_regular_file(opt.osrm_),
              "cannot find osrm dataset: [path={}]", opt.osrm_);
  utl::verify(boost::filesystem::is_regular_file(opt.osm_),
              "cannot find osm dataset: [path={}]", opt.osm_);

  utl::verify(opt.seq_cache_task_ == "ignore" ||
                  opt.seq_cache_task_ == "load" ||
                  opt.seq_cache_task_ == "dump",
              "invalid seq_cache_task");

  auto sequences = schedule_wrapper{opt.schedule_}.load_station_sequences();

  auto stations = collect_stations(sequences);
  find_stop_positions(opt.osm_, opt.schedule_, stations);
  filter_sequences(opt.filter_, sequences);
  auto const station_idx = make_station_index(sequences, std::move(stations));

  std::vector<resolved_station_seq> resolved_seqs;
  if (opt.seq_cache_task_ == "load") {
    resolved_seqs = read_from_fbs(opt.seq_cache_file_);
  } else {
    LOG(logging::info) << "processing " << sequences.size()
                       << " station sequences with "
                       << station_idx.stations_.size() << " unique stations.";

    auto routing = make_path_routing(station_idx, opt.osm_, opt.osrm_);

    resolved_seqs = resolve_sequences(sequences, routing);

    if (opt.seq_cache_task_ == "dump") {
      write_to_fbs(resolved_seqs, opt.seq_cache_file_);
    }
  }

  LOG(logging::info) << "post-processing " << resolved_seqs.size()
                     << " station sequences";

  auto post_graph = build_post_graph(std::move(resolved_seqs));
  post_process(post_graph);

  db_builder builder(opt.out_);
  builder.store_stations(station_idx.stations_);
  serialize_post_graph(post_graph, builder);
  builder.finish();
}

}  // namespace motis::path
