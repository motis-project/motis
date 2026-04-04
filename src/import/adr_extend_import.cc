#include "motis/import/adr_extend_import.h"

#include "cista/io.h"
#include "cista/memory_holder.h"

#include "adr/adr.h"
#include "adr/area_database.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "motis/adr_extend_tt.h"
#include "motis/data.h"

namespace motis {

namespace fs = std::filesystem;

adr_extend_import::adr_extend_import(fs::path const& data_path,
                                     config const& c,
                                     dataset_hashes const& h)
    : task{"adr_extend",
           data_path,
           c,
           {h.tt_,
            h.osm_,
            adr_version(),
            adr_ext_version(),
            n_version(),
            {"geocoding", c.geocoding_},
            {"reverse_geocoding", c.reverse_geocoding_}}} {}

adr_extend_import::~adr_extend_import() = default;

void adr_extend_import::run() {
  auto d = data{data_path_, false};
  d.load_tt("tt.bin");

  auto area_db = std::optional<adr::area_database>{};
  auto typeahead = cista::wrapped<adr::typeahead>{};

  if (fs::exists(data_path_ / "adr" / "t.bin")) {
    area_db.emplace(data_path_ / "adr", cista::mmap::protection::READ);
    typeahead = adr::read(data_path_ / "adr" / "t.bin");
  } else {
    typeahead = cista::wrapped<adr::typeahead>{
        cista::raw::make_unique<adr::typeahead>()};
  }

  auto const location_extra_place =
      adr_extend_tt(*d.tt_, area_db.has_value() ? &*area_db : nullptr,
                    *typeahead);

  auto ec = std::error_code{};
  fs::create_directories(data_path_ / "adr", ec);
  cista::write(data_path_ / "adr" / "t_ext.bin", *typeahead);
  cista::write(data_path_ / "adr" / "location_extra_place.bin",
               location_extra_place);

  auto r = adr::reverse{data_path_ / "adr", cista::mmap::protection::WRITE};
  r.build_rtree(*typeahead);
  r.write();
}

bool adr_extend_import::is_enabled() const {
  return c_.timetable_.has_value() &&
         (c_.geocoding_ || c_.reverse_geocoding_);
}

}  // namespace motis
