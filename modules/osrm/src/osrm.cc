#include "motis/osrm/osrm.h"

#include <mutex>

#include "boost/filesystem.hpp"

#include "tbb/task_scheduler_init.h"

#include "contractor/contractor.hpp"
#include "contractor/contractor_config.hpp"
#include "extractor/extractor.hpp"
#include "extractor/extractor_config.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/event_collector.h"

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

void osrm::import(motis::module::registry& reg) {
  for (auto const& p : profiles_) {
    event_collector{
        "osrm-" + boost::filesystem::path{p}.stem().generic_string(), reg,
        [&](std::map<MsgContent, msg_ptr> const& dependencies) {
          using import::OSMEvent;
          auto const osm =
              motis_content(OSMEvent, dependencies.at(MsgContent_OSMEvent));

          using namespace ::osrm::extractor;
          ExtractorConfig extr_conf;
          extr_conf.profile_path = p;
          extr_conf.requested_num_threads = std::thread::hardware_concurrency();
          extr_conf.generate_edge_lookup = false;
          extr_conf.small_component_size = 1000;
          extr_conf.input_path = osm->path()->str();
          extr_conf.UseDefaultOutputNames();
          Extractor{extr_conf}.run();

          using namespace ::osrm::contractor;
          ContractorConfig contr_conf;
          contr_conf.requested_num_threads =
              std::thread::hardware_concurrency();
          contr_conf.core_factor = 1.0;
          contr_conf.use_cached_priority = false;
          contr_conf.osrm_input_path = extr_conf.output_file_name;
          tbb::task_scheduler_init init(contr_conf.requested_num_threads);
          Contractor{contr_conf}.Run();
        }}
        .listen(MsgContent_OSMEvent);
  }
}

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
