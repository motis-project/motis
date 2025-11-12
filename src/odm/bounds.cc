#include "motis/odm/bounds.h"

#include "tg.h"

#include "fmt/std.h"

#include "utl/verify.h"

#include "cista/mmap.h"

namespace fs = std::filesystem;

namespace motis::odm {

bounds::bounds(fs::path const& p) {
  auto const f =
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ};

  geom_ = tg_parse_geojsonn(f.view().data(), f.size());

  if (tg_geom_error(geom_)) {
    char const* err = tg_geom_error(geom_);
    fmt::println("Error parsing ODM Bounds GeoJSON: {}", err);
    tg_geom_free(geom_);
    throw utl::fail("unable to parse {}: {}", p, err);
  }

  return;
}

bounds::~bounds() { tg_geom_free(geom_); }

bool bounds::contains(geo::latlng const& x) const {
  auto const point = tg_geom_new_point(tg_point{x.lng(), x.lat()});
  auto const result = tg_geom_within(point, geom_);
  tg_geom_free(point);
  return result;
}

}  // namespace motis::odm