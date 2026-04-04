#include "motis/import/osr_import.h"

#include "osr/extract/extract.h"

#include "motis/data.h"

namespace motis {

namespace fs = std::filesystem;

osr_import::osr_import(fs::path const& data_path,
                       config const& c,
                       dataset_hashes const& h)
    : task{"osr", data_path, c, {h.osm_, osr_version(), h.elevation_}} {}

osr_import::~osr_import() = default;

void osr_import::run() {
  auto const elevation_dir =
      c_.get_street_routing()
          .and_then([](config::street_routing const& sr) {
            return sr.elevation_data_dir_;
          })
          .value_or(fs::path{});
  osr::extract(true, fs::path{*c_.osm_}, data_path_ / "osr",
               elevation_dir);
}

bool osr_import::is_enabled() const { return c_.use_street_routing(); }

}  // namespace motis
