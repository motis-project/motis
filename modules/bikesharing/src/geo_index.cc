#include "motis/bikesharing/geo_index.h"

#include "boost/function_output_iterator.hpp"

#include "utl/to_vec.h"

#include "geo/point_rtree.h"

namespace motis::bikesharing {

struct geo_index::impl {
  explicit impl(database const& db) {
    auto const summary = db.get_summary();

    auto const& locations = summary.get()->terminals();
    rtree_ = std::make_unique<geo::point_rtree>(
        geo::make_point_rtree(*locations, [](auto const& loc) {
          return geo::latlng{loc->lat(), loc->lng()};
        }));
    terminal_ids_ = utl::to_vec(
        *locations, [](auto const& loc) { return loc->id()->str(); });
  }

  std::vector<close_terminal> get_terminals(double const lat, double const lng,
                                            double const radius) const {
    return utl::to_vec(
        rtree_->in_radius_with_distance({lat, lng}, radius),
        [this](auto const& result) {
          return close_terminal{terminal_ids_[result.second], result.first};
        });
  };

  std::unique_ptr<geo::point_rtree> rtree_;
  std::vector<std::string> terminal_ids_;
};

geo_index::geo_index(database const& db) : impl_(new impl(db)) {}

geo_index::~geo_index() = default;

std::vector<close_terminal> geo_index::get_terminals(
    double const lat, double const lng, double const radius) const {
  return impl_->get_terminals(lat, lng, radius);
}

}  // namespace motis::bikesharing
