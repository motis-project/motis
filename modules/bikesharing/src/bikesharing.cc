#include "motis/bikesharing/bikesharing.h"

#include <functional>
#include <numeric>

#include "motis/core/common/logging.h"
#include "motis/bikesharing/database.h"
#include "motis/bikesharing/error.h"
#include "motis/bikesharing/find_connections.h"
#include "motis/bikesharing/geo_index.h"
#include "motis/bikesharing/geo_terminals.h"
#include "motis/bikesharing/nextbike_initializer.h"

using namespace flatbuffers;
using namespace motis::logging;
using namespace motis::module;

namespace motis::bikesharing {

bikesharing::bikesharing() : module("Bikesharing Options", "bikesharing") {
  param(database_path_, "database_path",
        "bikesharing Database (folder or ':memory:')");
  param(nextbike_path_, "nextbike_path",
        "nextbike snapshots (folder or single file)");
  param(db_max_size_, "db_max_size", "virtual memory map size");
}

bikesharing::~bikesharing() = default;

void bikesharing::init(motis::module::registry& reg) {
  reg.subscribe("/init", std::bind(&bikesharing::init_module, this));
  reg.register_op("/bikesharing/search",
                  [this](msg_ptr const& req) { return search(req); });
  reg.register_op("/bikesharing/geo_terminals",
                  [this](msg_ptr const& req) { return geo_terminals(req); });
}

void bikesharing::init_module() {
  if (!database_path_.empty()) {
    database_ = std::make_unique<database>(database_path_, db_max_size_);

    if (database_->is_initialized()) {
      LOG(info) << "using initialized bikesharing database";
    } else {
      if (!nextbike_path_.empty()) {
        initialize_nextbike(nextbike_path_, *database_);
      }
    }

    if (database_->is_initialized()) {
      geo_index_ = std::make_unique<geo_index>(*database_);
    }
  }
}

msg_ptr bikesharing::search(msg_ptr const& req) const {
  ensure_initialized();

  using motis::bikesharing::BikesharingRequest;
  return motis::bikesharing::find_connections(
      *database_, *geo_index_, motis_content(BikesharingRequest, req));
}

msg_ptr bikesharing::geo_terminals(msg_ptr const& req) const {
  ensure_initialized();

  using motis::bikesharing::BikesharingGeoTerminalsRequest;
  return motis::bikesharing::geo_terminals(
      *database_, *geo_index_,
      motis_content(BikesharingGeoTerminalsRequest, req));
}

void bikesharing::ensure_initialized() const {
  if (!database_ || !geo_index_) {
    throw std::system_error(error::not_initialized);
  }
}

}  // namespace motis::bikesharing
