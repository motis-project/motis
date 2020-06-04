#include "motis/osrm/osrm.h"

#include <mutex>

#include "boost/filesystem.hpp"

#include "cista/reflection/comparable.h"

#include "contractor/contractor.hpp"
#include "contractor/contractor_config.hpp"
#include "extractor/extractor.hpp"
#include "extractor/extractor_config.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/clog_redirect.h"
#include "motis/module/context/motis_parallel_for.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/osrm/error.h"
#include "motis/osrm/router.h"

using namespace motis::module;
using namespace motis::logging;

namespace fs = boost::filesystem;

namespace motis::osrm {

struct import_state {
  CISTA_COMPARABLE()
  named<std::string, MOTIS_NAME("path")> path_;
  named<cista::hash_t, MOTIS_NAME("hash")> hash_;
  named<size_t, MOTIS_NAME("size")> size_;
};

osrm::osrm() : module("OSRM Options", "osrm") {
  param(profiles_, "profiles", "lua profile paths");
}

osrm::~osrm() = default;

void osrm::import(motis::module::registry& reg) {
  for (auto const& p : profiles_) {
    auto const profile_name =
        boost::filesystem::path{p}.stem().generic_string();
    std::make_shared<event_collector>(
        get_data_directory().generic_string(), "osrm-" + profile_name, reg,
        [this, profile_name,
         p](std::map<std::string, msg_ptr> const& dependencies) {
          using import::OSMEvent;
          using namespace ::osrm::extractor;
          using namespace ::osrm::contractor;

          auto const osm = motis_content(OSMEvent, dependencies.at("OSM"));
          auto const state = import_state{data_path(osm->path()->str()),
                                          osm->hash(), osm->size()};

          auto const osm_stem = fs::path{fs::path{osm->path()->str()}.stem()}
                                    .stem()
                                    .generic_string();

          auto const dir = get_data_directory() / "osrm" / profile_name;
          fs::create_directories(dir);

          ExtractorConfig extr_conf;
          extr_conf.profile_path = p;
          extr_conf.requested_num_threads = std::thread::hardware_concurrency();
          extr_conf.generate_edge_lookup = false;
          extr_conf.small_component_size = 1000;
          extr_conf.input_path = osm->path()->str();
          extr_conf.UseDefaultOutputNames((dir / osm_stem).generic_string());
          if (read_ini<import_state>(dir / "import.ini") != state) {
            Extractor{extr_conf}.run();

            ContractorConfig contr_conf;
            contr_conf.requested_num_threads =
                std::thread::hardware_concurrency();
            contr_conf.core_factor = 1.0;
            contr_conf.use_cached_priority = false;
            contr_conf.osrm_input_path = extr_conf.output_file_name;
            contr_conf.UseDefaultOutputNames();
            Contractor{contr_conf}.Run();

            write_ini(dir / "import.ini", state);
          }

          datasets_.emplace_back(extr_conf.output_file_name);

          message_creator fbb;
          fbb.create_and_finish(
              MsgContent_OSRMEvent,
              motis::import::CreateOSRMEvent(
                  fbb, fbb.CreateString(extr_conf.output_file_name),
                  fbb.CreateString(profile_name))
                  .Union(),
              "/import", DestinationType_Topic);
          motis_publish(make_msg(fbb));
        })
        ->require("OSM", [](msg_ptr const& msg) {
          return msg->get()->content_type() == MsgContent_OSMEvent;
        });
  }
}

bool osrm::import_successful() const {
  return datasets_.size() == profiles_.size();
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
