#pragma once

#include <filesystem>
#include <optional>

#include "cista/containers/pair.h"
#include "cista/containers/vector.h"

#include "lmdb/lmdb.hpp"

#include "geo/box.h"

#include "nigiri/types.h"

#include "osr/routing/profile.h"

#include "motis/config.h"
#include "motis/fwd.h"

namespace motis {

struct shape_cache_entry {
  bool valid() const {
    return shape_idx_ != nigiri::scoped_shape_idx_t::invalid();
  }

  nigiri::scoped_shape_idx_t shape_idx_{nigiri::scoped_shape_idx_t::invalid()};
  cista::offset::vector<nigiri::shape_offset_t> offsets_;
  geo::box route_bbox_;
  cista::offset::vector<geo::box> segment_bboxes_;
};

using shape_cache_key = cista::offset::pair<osr::search_profile,
                                            cista::offset::vector<geo::latlng>>;

struct shape_cache {
  explicit shape_cache(std::filesystem::path const&,
                       mdb_size_t = sizeof(void*) >= 8
                                        ? 256ULL * 1024ULL * 1024ULL * 1024ULL
                                        : 256U * 1024U * 1024U);

  std::optional<shape_cache_entry> get(shape_cache_key const&);
  void put(shape_cache_key const&, shape_cache_entry const&);

  lmdb::env env_;
};

void route_shapes(osr::ways const&,
                  osr::lookup const&,
                  nigiri::timetable&,
                  nigiri::shapes_storage&,
                  config::timetable::route_shapes const&,
                  std::array<bool, nigiri::kNumClasses> const&,
                  shape_cache*);

}  // namespace motis
