#pragma once

#include <memory>

#include "tg.h"

#include "osr/types.h"

#include "nigiri/types.h"

#include "motis/fwd.h"
#include "motis/gbfs/compression.h"
#include "motis/types.h"

namespace motis::flex {

struct flex_areas {
  explicit flex_areas(nigiri::timetable const&,
                      osr::ways const&,
                      osr::lookup const&);
  ~flex_areas();

  void add_area(nigiri::flex_area_idx_t,
                osr::bitvec<osr::node_idx_t>&,
                osr::bitvec<osr::node_idx_t>& tmp) const;

  bool is_in_area(nigiri::flex_area_idx_t, geo::latlng const&) const;

  vector_map<nigiri::flex_area_idx_t, gbfs::compressed_bitvec> area_nodes_;
  vector_map<nigiri::flex_area_idx_t, tg_geom*> idx_;
};

}  // namespace motis::flex