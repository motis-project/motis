#include "motis/import/adr_import.h"

#include "adr/adr.h"
#include "adr/area_database.h"
#include "adr/formatter.h"
#include "adr/typeahead.h"

#include "motis/data.h"

namespace motis {

void adr_import::load() {
  d_.t_ = adr::read(data_path_ / "adr" / "t.bin");
  d_.tc_ = std::make_unique<adr::cache>(d_.t_->strings_.size(), 100U);
  d_.f_ = std::make_unique<adr::formatter>();

  if (c_.reverse_geocoding_) {
    d_.load_reverse_geocoder();
  }
}

void adr_import::unload() { d_.t_ = {}; }

void adr_import::run() {
  adr::extract(c_.osm_.value(), data_path_ / "adr", data_path_ / "adr");

  // We can't use d.load_geocoder() here because
  // adr_extend expects the base-line version
  // without extra timetable information.
  d_.t_ = adr::read(data_path_ / "adr" / "t.bin");
  d_.tc_ = std::make_unique<adr::cache>(d_.t_->strings_.size(), 100U);

  if (c_.reverse_geocoding_) {
    d_.load_reverse_geocoder();
  }

  d_.tc_ = std::make_unique<adr::cache>(d_.t_->strings_.size(), 100U);
  d_.f_ = std::make_unique<adr::formatter>();
}

bool adr_import::is_enabled() const {
  return c_.geocoding_ || c_.reverse_geocoding_;
}

}  // namespace motis