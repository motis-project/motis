#include "motis/import/tiles_import.h"

#include "tiles/db/clear_database.h"
#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/feature_pack.h"
#include "tiles/db/pack_file.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/osm/load_coastlines.h"
#include "tiles/osm/load_osm.h"

#include "utl/progress_tracker.h"

#include "motis/data.h"

namespace motis {

namespace fs = std::filesystem;

tiles_import::tiles_import(fs::path const& data_path,
                           config const& c,
                           dataset_hashes const& h)
    : task{"tiles", data_path, c, {tiles_version(), h.osm_, h.tiles_}} {}

tiles_import::~tiles_import() = default;

void tiles_import::run() {
  auto const progress_tracker = utl::get_active_progress_tracker();
  auto const dir = data_path_ / "tiles";
  auto const path = (dir / "tiles.mdb").string();

  auto ec = std::error_code{};
  fs::create_directories(data_path_ / "tiles", ec);

  progress_tracker->status("Clear Database");
  ::tiles::clear_database(path, c_.tiles_->db_size_);
  ::tiles::clear_pack_file(path.c_str());

  auto db_env = ::tiles::make_tile_database(path.c_str(), c_.tiles_->db_size_);
  ::tiles::tile_db_handle db_handle{db_env};
  ::tiles::pack_handle pack_handle{path.c_str()};

  {
    ::tiles::feature_inserter_mt inserter{
        ::tiles::dbi_handle{db_handle, db_handle.features_dbi_opener()},
        pack_handle};

    if (c_.tiles_->coastline_.has_value()) {
      progress_tracker->status("Load Coastlines").out_bounds(0, 20);
      ::tiles::load_coastlines(db_handle, inserter,
                               c_.tiles_->coastline_->generic_string());
    }

    progress_tracker->status("Load Features").out_bounds(20, 70);
    ::tiles::load_osm(db_handle, inserter, c_.osm_->generic_string(),
                      c_.tiles_->profile_.generic_string(),
                      dir.generic_string(), c_.tiles_->flush_threshold_);
  }

  progress_tracker->status("Pack Features").out_bounds(70, 90);
  ::tiles::pack_features(db_handle, pack_handle);

  progress_tracker->status("Prepare Tiles").out_bounds(90, 100);
  ::tiles::prepare_tiles(db_handle, pack_handle, 10);
}

bool tiles_import::is_enabled() const { return c_.tiles_.has_value(); }

}  // namespace motis
