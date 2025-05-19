#include "motis/flex/flex_areas.h"

#include "osr/lookup.h"

#include "utl/concat.h"
#include "utl/parallel_for.h"

#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis::flex {

tg_ring* convert_ring(std::vector<tg_point>& ring_tmp, auto&& osm_ring) {
  ring_tmp.clear();
  for (auto const& p : osm_ring) {
    ring_tmp.emplace_back(tg_point{p.lng_, p.lat_});
  }
  return tg_ring_new(ring_tmp.data(), static_cast<int>(ring_tmp.size()));
}

flex_areas::~flex_areas() {
  for (auto const& mp : idx_) {
    tg_geom_free(mp);
  }
}

flex_areas::flex_areas(nigiri::timetable const& tt,
                       osr::ways const& w,
                       osr::lookup const& l) {
  struct tmp {
    std::vector<tg_point> ring_tmp_;
    std::vector<tg_ring*> inner_tmp_;
    std::vector<tg_poly*> polys_tmp_;
    basic_string<n::flex_area_idx_t> areas_;
  };

  area_nodes_.resize(tt.flex_area_outers_.size());
  idx_.resize(tt.flex_area_outers_.size());
  utl::parallel_for_run_threadlocal<tmp>(
      tt.flex_area_outers_.size(), [&](tmp& tmp, std::size_t const i) {
        tmp.polys_tmp_.clear();

        auto const a = n::flex_area_idx_t{i};
        auto const& outer_rings = tt.flex_area_outers_[a];

        auto box = geo::box{};
        for (auto const [outer_idx, outer_ring] : utl::enumerate(outer_rings)) {
          tmp.inner_tmp_.clear();
          for (auto const inner_ring :
               tt.flex_area_inners_[a][static_cast<unsigned>(outer_idx)]) {
            tmp.inner_tmp_.emplace_back(
                convert_ring(tmp.ring_tmp_, inner_ring));
          }

          for (auto const& c : outer_ring) {
            box.extend(c);
          }

          auto const outer = convert_ring(tmp.ring_tmp_, outer_ring);
          auto const poly =
              tg_poly_new(outer, tmp.inner_tmp_.data(),
                          static_cast<int>(tmp.inner_tmp_.size()));
          tg_ring_free(outer);
          tmp.polys_tmp_.emplace_back(poly);
        }

        idx_[a] = tg_geom_new_multipolygon(
            tmp.polys_tmp_.data(), static_cast<int>(tmp.polys_tmp_.size()));
        auto b = osr::bitvec<osr::node_idx_t>{};
        b.resize(w.n_nodes());
        l.find(tt.flex_area_bbox_[a], [&](osr::way_idx_t const way) {
          for (auto const& x : w.r_->way_nodes_[way]) {
            if (is_in_area(a, w.get_node_pos(x).as_latlng())) {
              b.set(x, true);
            }
          }
        });

        area_nodes_[a] = gbfs::compress_bitvec(b);

        for (auto const& x : tmp.inner_tmp_) {
          tg_ring_free(x);
        }

        for (auto const x : tmp.polys_tmp_) {
          tg_poly_free(x);
        }
      });
}

bool flex_areas::is_in_area(nigiri::flex_area_idx_t const a,
                            geo::latlng const& c) const {
  auto const point = tg_geom_new_point(tg_point{c.lng(), c.lat()});
  auto const result = tg_geom_within(point, idx_[a]);
  tg_geom_free(point);
  return result;
}

void flex_areas::add_area(nigiri::flex_area_idx_t a,
                          osr::bitvec<osr::node_idx_t>& b,
                          osr::bitvec<osr::node_idx_t>& tmp) const {
  gbfs::decompress_bitvec(area_nodes_[a], tmp);
  tmp.for_each_set_bit([&](auto&& i) { b.set(osr::node_idx_t{i}, true); });
}

}  // namespace motis::flex