#include "motis/import/adr_import.h"

#include "adr/adr.h"
#include "adr/area_database.h"
#include "adr/formatter.h"
#include "adr/typeahead.h"

#include "motis/data.h"

namespace motis {

namespace fs = std::filesystem;

adr_import::adr_import(fs::path const& data_path,
                       config const& c,
                       dataset_hashes const& h)
    : task{"adr", data_path, c, {h.osm_, adr_version()}} {}

adr_import::~adr_import() = default;

void adr_import::run() {
  if (!c_.osm_.has_value()) {
    return;
  }
  adr::extract(c_.osm_.value(), data_path_ / "adr", data_path_ / "adr");
}

bool adr_import::is_enabled() const {
  return c_.geocoding_ || c_.reverse_geocoding_;
}

}  // namespace motis
