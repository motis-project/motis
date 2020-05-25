#pragma once

#include "utl/to_vec.h"

#include "motis/raptor/mark_store.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_util.h"

namespace motis::raptor {

struct additional_start {
  additional_start() = delete;
  additional_start(station_id const s_id, time const offset)
      : s_id_(s_id), offset_(offset) {}
  station_id s_id_{invalid<station_id>};
  time offset_{invalid<time>};
};

auto inline get_add_starts(
    raptor_schedule const& raptor_sched,
    raptor_timetable const&,  // TODO IMPLEMENT START FOOT
    station_id const source, bool const use_start_footpaths,
    bool const use_meta_starts) {
  std::vector<additional_start> add_starts;

  if (use_start_footpaths) {
    auto const& init_footpaths = raptor_sched.initialization_footpaths_[source];
    append_vector(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                    return additional_start(f.to_, f.duration_);
                  }));
  }

  if (use_meta_starts) {
    auto const& metas = raptor_sched.equivalent_stations_[source];
    append_vector(add_starts, utl::to_vec(metas, [](auto const s_id) {
                    return additional_start(s_id, 0);
                  }));

    // Footpaths from meta stations
    if (use_start_footpaths) {
      for (auto const meta : metas) {
        auto const& init_footpaths =
            raptor_sched.initialization_footpaths_[meta];
        append_vector(add_starts, utl::to_vec(init_footpaths, [](auto const f) {
                        return additional_start(f.to_, f.duration_);
                      }));
      }
    }
  }

  return add_starts;
}

struct base_query {
  station_id source_{invalid<station_id>};
  station_id target_{invalid<station_id>};

  time source_time_begin_{invalid<time>};
  time source_time_end_{invalid<time>};

  bool forward_{true};
  bool use_dest_metas_{true};
};

struct raptor_query : base_query {
  raptor_query() = delete;
  raptor_query(raptor_query const&) = delete;

  raptor_query(base_query const& bq, raptor_schedule const& raptor_sched,
               raptor_timetable const& tt, bool const use_start_footpaths,
               bool const use_start_metas)
      : tt_(tt) {
    static_cast<base_query&>(*this) = bq;

    result_ = std::make_unique<raptor_result>(tt_.stop_count());

    add_starts_ = get_add_starts(raptor_sched, tt, source_, use_start_footpaths,
                                 use_start_metas);
  }

  raptor_timetable const& tt_;
  std::vector<additional_start> add_starts_;
  std::unique_ptr<raptor_result> result_;
};

}  // namespace motis::raptor