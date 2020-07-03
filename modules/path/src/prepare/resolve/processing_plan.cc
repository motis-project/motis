#include "motis/path/prepare/resolve/processing_plan.h"

#include <tuple>

#include "geo/box.h"
#include "geo/latlng.h"

#include "utl/get_or_create.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/resolve/path_routing.h"
#include "motis/path/prepare/schedule/station_sequences.h"

namespace ml = motis::logging;

namespace motis::path {

processing_plan make_processing_plan(
    path_routing& routing, mcd::vector<station_seq> const& sequences) {
  ml::scoped_timer t{"make_processing_plan"};
  processing_plan pp;
  {
    mcd::hash_map<part_task_key, size_t> part_task_map;
    for (auto const& seq : sequences) {
      foreach_path_category(seq.classes_, [&](auto const& path_cat,
                                              auto motis_cats) {
        auto s_task_idx = pp.seq_tasks_.size();
        auto& s_task = pp.seq_tasks_.emplace_back(&seq, std::move(motis_cats));

        if (path_cat == source_spec::category::UNKNOWN) {
          return;
        }

        for (auto* strategy : routing.strategies_for(path_cat)) {
          for (auto i = 0UL; i < seq.station_ids_.size() - 1; ++i) {
            auto const& station_id_from = seq.station_ids_[i];
            auto const& station_id_to = seq.station_ids_[i + 1];

            auto const make_task = [&](auto const& from, auto const& to) {
              part_task_key key{strategy, from.str(), to.str()};

              auto p_task_idx = utl::get_or_create(part_task_map, key, [&] {
                geo::box box{seq.coordinates_[i], seq.coordinates_[i + 1]};
                pp.part_tasks_.emplace_back(geo::tile_hash_32(box.min_), key);
                return pp.part_tasks_.size() - 1;
              });

              s_task.part_dependencies_.push_back(p_task_idx);
              pp.part_tasks_[p_task_idx].seq_dependencies_.push_back(
                  s_task_idx);
            };

            if (i != 0) {
              make_task(station_id_from, station_id_from);  // within station
            }
            make_task(station_id_from, station_id_to);  // between stations
          }
        }
      });
    }
  }

  LOG(ml::info) << "processing_plan: have " << pp.seq_tasks_.size()
                << " sequences with " << pp.part_tasks_.size() << " parts";

  {
    auto tmp = utl::to_vec(pp.part_tasks_, [index = 0U](auto const& t) mutable {
      return std::pair<uint32_t, uint32_t>{t.location_hash_, index++};
    });

    std::sort(begin(tmp), end(tmp), [](auto const& lhs, auto const& rhs) {
      return lhs.first < rhs.first;
    });

    pp.part_task_queue_ =
        utl::to_vec(tmp, [](auto const& pair) { return pair.second; });
  }

  return pp;
}

}  // namespace motis::path
