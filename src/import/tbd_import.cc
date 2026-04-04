#include "motis/import/tbd_import.h"

#include "cista/io.h"

#include "nigiri/routing/tb/preprocess.h"

#include "motis/data.h"

namespace n = nigiri;

namespace motis {

namespace fs = std::filesystem;

tbd_import::tbd_import(fs::path const& data_path,
                       config const& c,
                       dataset_hashes const& h)
    : task{"tbd", data_path, c, {h.tt_, n_version(), tbd_version()}} {}

tbd_import::~tbd_import() = default;

void tbd_import::run() {
  auto d = data{data_path_, false};
  d.load_tt("tt.bin");

  cista::write(data_path_ / "tbd.bin",
               n::routing::tb::preprocess(*d.tt_, n::kDefaultProfile));
}

bool tbd_import::is_enabled() const {
  return c_.timetable_.has_value() && c_.timetable_->tb_;
}

}  // namespace motis
