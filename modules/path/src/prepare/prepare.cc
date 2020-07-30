#include "motis/path/prepare/prepare.h"

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include "boost/filesystem.hpp"

#include "cista/memory_holder.h"

#include "conf/configuration.h"

#include "geo/polygon.h"

#include "utl/concat.h"
#include "utl/erase_if.h"
#include "utl/verify.h"
#include "utl/zip.h"

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

namespace fs = boost::filesystem;
namespace ml = motis::logging;

namespace motis::path {

inline void filter_sequences(std::vector<std::string> const& filters,
                             mcd::vector<station_seq>& sequences) {
  ml::scoped_timer timer("filter station sequences");
  for (auto const& filter : filters) {
    auto const pos = filter.find_first_of(':');
    utl::verify(pos != std::string::npos, "unexpected filter");
    auto const key = filter.substr(0, pos);
    auto const value = filter.substr(pos + 1);

    if (key == "id") {
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::none_of(begin(seq.station_ids_), end(seq.station_ids_),
                            [&](auto const& id) { return id == value; });
      });
    } else if (key == "seq") {
      std::vector<std::string> ids;
      boost::split(ids, value, boost::is_any_of("."));
      utl::erase_if(sequences, [&ids](auto const& seq) {
        return !std::equal(begin(ids), end(ids), begin(seq.station_ids_),
                           end(seq.station_ids_));
      });
    } else if (key == "extent") {
      utl::verify(boost::filesystem::is_regular_file(value),
                  "cannot find extent polygon");
      auto const extent_polygon = geo::read_poly_file(value);
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::any_of(begin(seq.coordinates_), end(seq.coordinates_),
                           [&](auto const& coord) {
                             return !geo::within(coord, extent_polygon);
                           });
      });
    } else if (key == "limit") {
      sequences.resize(std::min(static_cast<size_t>(std::stoul(value)),
                                static_cast<size_t>(sequences.size())));
    } else if (key == "cat") {
      auto clasz = static_cast<service_class>(std::stoi(value));
      utl::erase_if(sequences, [&](auto const& seq) {
        return std::find(begin(seq.classes_), end(seq.classes_), clasz) ==
               end(seq.classes_);
      });
      for (auto& seq : sequences) {
        seq.classes_ = {clasz};
      }
    } else {
      LOG(ml::info) << "unknown filter: " << key;
    }
  }
}

struct cachable_step {
  explicit cachable_step(std::string const& fname, std::string const& task)
      : fname_{fname}, task_{task} {
    utl::verify(task_ == "ignore" || task_ == "load" || task_ == "dump" ||
                    task_ == "use",
                "cachable_step: invalid task {}", task_);
  }

  template <typename LoadFn, typename DumpFn, typename ComputeFn>
  auto get(LoadFn&& load, DumpFn&& dump, ComputeFn&& compute)
      -> decltype(compute()) {
    if (task_ == "load" || (task_ == "use" && fs::is_regular_file(fname_))) {
      return load(fname_);
    } else {
      auto value = compute();
      if (task_ == "dump" || task_ == "use") {
        dump(fname_, value);
      }
      return value;
    }
  }

  std::string const& fname_;
  std::string const& task_;
};

void prepare(prepare_settings const& opt) {
  utl::verify(fs::is_regular_file(opt.osrm_),
              "cannot find osrm dataset: [path={}]", opt.osrm_);
  utl::verify(fs::is_regular_file(opt.osm_),
              "cannot find osm dataset: [path={}]", opt.osm_);

  cachable_step osm_cache{opt.osm_cache_file_, opt.osm_cache_task_};
  cachable_step seq_cache{opt.seq_cache_file_, opt.seq_cache_task_};

  auto progress_tracker = utl::get_active_progress_tracker();

  cista::memory_holder osm_data_mem;
  mcd::unique_ptr<osm_data> osm_data_ptr;
  auto const load_osm_data = [&] {
    osm_data_ptr = osm_cache.get(
        [&](auto const& f) { return read_osm_data(f, osm_data_mem); },
        [&](auto const& f, auto const& dat) { return write_osm_data(f, dat); },
        [&] { return parse_osm(opt.osm_); });
  };

  cista::memory_holder seq_mem;
  auto resolved_seqs = seq_cache.get(
      [&](auto const& f) { return read_station_sequences(f, seq_mem); },
      [&](auto const& f, auto const& rs) { write_station_sequences(f, rs); },
      [&] {
        progress_tracker->status("Load Station Sequences").out_bounds(0, 5);
        mcd::vector<station_seq> sequences;
        if (opt.schedules_.size() == 1 && opt.prefixes_.empty()) {
          sequences =
              schedule_wrapper{opt.schedules_[0]}.load_station_sequences("");
        } else {
          for (auto const& [sched, prefix] :
               utl::zip(opt.schedules_, opt.prefixes_)) {
            utl::concat(sequences,
                        schedule_wrapper{sched}.load_station_sequences(prefix));
          }
        }

        filter_sequences(opt.filter_, sequences);
        utl::verify(!sequences.empty(), "sequences empty (nothing to do)!");

        load_osm_data();  // load only if resolve_sequences runs!
        LOG(ml::info) << "OSM DATA: " << osm_data_ptr->stop_positions_.size()
                      << " stop positions, " << osm_data_ptr->plattforms_.size()
                      << " plattforms, " << osm_data_ptr->profiles_.size()
                      << " profiles";

        progress_tracker->status("Prepare Stations");
        auto stations = load_stations(sequences, *osm_data_ptr);

        LOG(ml::info) << "processing " << sequences.size()
                      << " station sequences with " << stations.stations_.size()
                      << " unique stations.";

        progress_tracker->status("Make Path Routing");
        auto routing =
            make_path_routing(sequences, stations, *osm_data_ptr, opt.osrm_);

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
