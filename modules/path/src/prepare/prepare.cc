#include "motis/path/prepare/prepare.h"

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/filesystem.hpp"

#include "cista/memory_holder.h"

#include "conf/configuration.h"

#include "geo/polygon.h"

#include "utl/erase_if.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"

#include "motis/path/path_database.h"
#include "motis/path/prepare/db_builder.h"
#include "motis/path/prepare/osm/osm_data.h"
#include "motis/path/prepare/post/build_post_graph.h"
#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/post/post_processor.h"
#include "motis/path/prepare/post/post_serializer.h"
#include "motis/path/prepare/resolve/path_routing.h"
#include "motis/path/prepare/resolve/resolve_sequences.h"
#include "motis/path/prepare/schedule/schedule_wrapper.h"
#include "motis/path/prepare/schedule/station_sequences.h"
#include "motis/path/prepare/schedule/stations.h"

namespace ml = motis::logging;

namespace motis::path {

inline void filter_sequences(std::vector<std::string> const& filters,
                             mcd::vector<station_seq>& sequences) {
  ml::scoped_timer timer("filter station sequences");
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
        return !std::equal(begin(ids), end(ids), begin(seq.station_ids_),
                           end(seq.station_ids_));
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
      sequences.resize(std::min(static_cast<size_t>(std::stoul(tokens[1])),
                                static_cast<size_t>(sequences.size())));
    } else if (tokens[0] == "cat") {
      auto clasz = static_cast<service_class>(std::stoi(tokens[1]));
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::find(begin(seq.classes_), end(seq.classes_), clasz) ==
               end(seq.classes_);
      });
      for (auto& seq : sequences) {
        seq.classes_ = {clasz};
      }
    } else {
      LOG(ml::info) << "unknown filter: " << tokens[0];
    }
  }
}

struct cachable_step {
  explicit cachable_step(std::string const& task) : task_{task} {
    utl::verify(task_ == "ignore" || task_ == "load" || task_ == "dump",
                "cachable_step: invalid task {}", task_);
  }

  template <typename LoadFn, typename DumpFn, typename ComputeFn>
  auto get(LoadFn&& load, DumpFn&& dump, ComputeFn&& compute)
      -> decltype(compute()) {
    if (task_ == "load") {
      return load();
    } else {
      auto value = compute();
      if (task_ == "dump") {
        dump(value);
      }
      return value;
    }
  }

  std::string const& task_;
};

void prepare(prepare_settings const& opt) {
  utl::verify(boost::filesystem::is_regular_file(opt.osrm_),
              "cannot find osrm dataset: [path={}]", opt.osrm_);
  utl::verify(boost::filesystem::is_regular_file(opt.osm_),
              "cannot find osm dataset: [path={}]", opt.osm_);

  cachable_step osm_cache{opt.osm_cache_task_};
  cachable_step seq_cache{opt.seq_cache_task_};

  auto progress_tracker = utl::get_active_progress_tracker();

  cista::memory_holder osm_data_mem;
  mcd::unique_ptr<osm_data> osm_data_ptr;
  auto const load_osm_data = [&] {
    osm_data_ptr = osm_cache.get(
        [&] { return read_osm_data(opt.osm_cache_file_, osm_data_mem); },
        [&](auto const& d) { return write_osm_data(opt.osm_cache_file_, d); },
        [&] { return parse_osm(opt.osm_); });
  };

  cista::memory_holder seq_mem;
  auto resolved_seqs = seq_cache.get(
      [&] { return read_station_sequences(opt.seq_cache_file_, seq_mem); },
      [&](auto const& rs) { write_station_sequences(opt.seq_cache_file_, rs); },
      [&] {
        progress_tracker->status("Load Station Sequences").out_bounds(0, 5);
        auto sequences =
            schedule_wrapper{opt.schedule_}.load_station_sequences();
        filter_sequences(opt.filter_, sequences);

        load_osm_data();  // load only if resolve_sequences runs!
        LOG(ml::info) << "OSM DATA: " << osm_data_ptr->stop_positions_.size()
                      << " stop positions, " << osm_data_ptr->plattforms_.size()
                      << " plattforms, " << osm_data_ptr->profiles_.size()
                      << " profiles";

        progress_tracker->status("Prepare Stations");
        auto stations = collect_stations(sequences);
        annotate_stop_positions(*osm_data_ptr, stations);

        LOG(ml::info) << "processing " << sequences.size()
                      << " station sequences with " << stations.stations_.size()
                      << " unique stations.";

        progress_tracker->status("Make Path Routing");
        auto routing = make_path_routing(stations, *osm_data_ptr, opt.osrm_);

        progress_tracker->status("Resolve Sequences").out_bounds(25, 90);
        return mcd::make_unique<mcd::vector<station_seq>>(
            resolve_sequences(sequences, routing));
      });

  LOG(ml::info) << "post-processing " << resolved_seqs->size()
                << " station sequences";

  progress_tracker->status("Post Processing");
  auto post_graph = build_post_graph(std::move(resolved_seqs));
  post_process(post_graph);

  db_builder builder(opt.out_);
  builder.store_stations(*post_graph.originals_);
  serialize_post_graph(post_graph, builder);
  builder.finish();
}

}  // namespace motis::path
