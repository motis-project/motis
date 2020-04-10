#include "motis/osrm/osrm.h"

#include <mutex>

#include "boost/filesystem.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_parallel_for.h"

#include "motis/osrm/error.h"
#include "motis/osrm/router.h"

using namespace motis::module;
using namespace motis::logging;

namespace fs = boost::filesystem;

namespace motis::osrm {

osrm::osrm() : module("OSRM Options", "osrm") {
  param(datasets_, "dataset",
        ".osrm file (multiple datasets accessible by folder name)");
}

osrm::~osrm() = default;

void osrm::init(motis::module::registry& reg) {
  reg.subscribe("/init", [this] { init_async(); });
  reg.register_op("/osrm/one_to_many", [this](msg_ptr const& msg) {
    auto const req = motis_content(OSRMOneToManyRequest, msg);
    return get_router(req->profile()->str())->one_to_many(req);
  });
  reg.register_op("/osrm/via", [this](msg_ptr const& msg) {
    auto const req = motis_content(OSRMViaRouteRequest, msg);
    return get_router(req->profile()->str())->via(req);
  });
  reg.register_op("/osrm/smooth_via", [this](msg_ptr const& msg) {
    auto const req = motis_content(OSRMSmoothViaRouteRequest, msg);
    return get_router(req->profile()->str())->smooth_via(req);
  });
}

void osrm::init_async() {
  std::mutex mutex;
  motis_parallel_for(
      datasets_, ([&mutex, this](std::string const& dataset) {
        fs::path path(dataset);
        auto directory = path.parent_path();
        if (!is_directory(directory)) {
          throw std::runtime_error("OSRM dataset is not a folder!");
        }

        auto const profile = directory.filename().string();
        scoped_timer timer("loading OSRM dataset: " + profile);
        auto r = std::make_unique<router>(dataset);

        std::lock_guard<std::mutex> lock(mutex);
        routers_.emplace(profile, std::move(r));
      }));
}

router const* osrm::get_router(std::string const& profile) {
  auto const it = routers_.find(profile);
  if (it == end(routers_)) {
    throw std::system_error(error::profile_not_available);
  }
  return it->second.get();
}

}  // namespace motis::osrm
