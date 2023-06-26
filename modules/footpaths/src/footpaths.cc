#include "motis/footpaths/footpaths.h"

#include <filesystem>
#include <iostream>

#include "cista/memory_holder.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"

using namespace motis::logging;
using namespace motis::module;

namespace fs = std::filesystem;

namespace motis::footpaths {

// load already known/stored data
struct import_state {
  CISTA_COMPARABLE();
  // import nigiri state
  named<cista::hash_t, MOTIS_NAME("nigiri_hash")> nigiri_hash_;

  // import osm state
  named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  named<cista::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  named<size_t, MOTIS_NAME("osm_size")> osm_size_;

  // import ppr state
  named<std::string, MOTIS_NAME("ppr_graph_path")> ppr_graph_path_;
  named<cista::hash_t, MOTIS_NAME("ppr_graph_hash")> ppr_graph_hash_;
  named<size_t, MOTIS_NAME("ppr_graph_size")> ppr_graph_size_;
  named<cista::hash_t, MOTIS_NAME("ppr_profiles_hash")> ppr_profiles_hash_;
  named<int, MOTIS_NAME("max_walk_duration")> max_walk_duration_;
};

struct footpaths::impl {
  nigiri::timetable tt_;
};

footpaths::footpaths() : module("Footpaths", "footpaths") {
  param(max_walk_duration_, "max_walk_duration",
        "Maximum walking time per path in minutes.");
}

footpaths::~footpaths() = default;

fs::path footpaths::module_data_dir() const {
  return get_data_directory() / "footpaths";
}

void footpaths::init(motis::module::registry& reg) { std::ignore = reg; }

void footpaths::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "footpaths", reg,
      [this](event_collector::dependencies_map_t const& dependencies,
             event_collector::publish_fn_t const&) {
        using import::NigiriEvent;
        using import::OSMEvent;
        using import::PPREvent;

        impl_ = std::make_unique<impl>();
        impl_->tt_ = *get_shared_data<nigiri::timetable*>(
            to_res_id(global_res_id::NIGIRI_TIMETABLE));

        auto const nigiri_event =
            motis_content(NigiriEvent, dependencies.at("NIGIRI"));
        LOG(info) << "hashes: nigiri=" << nigiri_event->hash();
        auto const osm_event = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const ppr_event = motis_content(PPREvent, dependencies.at("PPR"));
        auto const state =
            import_state{nigiri_event->hash(),
                         data_path(osm_event->path()->str()),
                         osm_event->hash(),
                         osm_event->size(),
                         data_path(ppr_event->graph_path()->str()),
                         ppr_event->graph_hash(),
                         ppr_event->graph_size(),
                         ppr_event->profiles_hash(),
                         max_walk_duration_};

        // verify that data structure in Nigiri was adjusted to necessary number
        // of profiles
        uint16_t const& no_profiles = ppr_event->profiles()->size();
        utl::verify(
            no_profiles == impl_->tt_.locations_.footpaths_out_.size(),
            "[footpath_out_] Profiles are not fully initialized. "
            "(IS/SHOULD): " +
                std::to_string(impl_->tt_.locations_.footpaths_out_.size()) +
                "/" + std::to_string(no_profiles));
        utl::verify(
            no_profiles == impl_->tt_.locations_.footpaths_in_.size(),
            "[footpath_in_] Profiles are not fully initialized. "
            "(IS/SHOULD): " +
                std::to_string(impl_->tt_.locations_.footpaths_out_.size()) +
                "/" + std::to_string(no_profiles));

        // load ppr profiles; store all profiles in a map
        // (profile-name, profile-info)
        motis::ppr::read_profile_files(
            utl::to_vec(*ppr_event->profiles(),
                        [](auto const& p) { return p->name()->str(); }),
            ppr_profiles_);

        // generate (profile_name, profile position in footpath)
        // REMARK: there is no need to use the default profile
        for (auto& p : ppr_profiles_) {
          p.second.profile_.duration_limit_ = max_walk_duration_ * 60;
          ppr_profile_pos_.insert({p.first, ppr_profile_pos_.size()});
        }

        // Implementation of footpaths is inspired by parking

        // TODO (Carsten, 1) Calculate internal and external transfers
        // TODO (Carsten, 1) Use all known ppr-profiles to update footpaths

        // TODO (Carsten, 2) Check for existing calculations. if state ==
        // import-state: load existing, otherwise: recalculate
        // ppr-profiles changed? nigiri-graph changed?

        /**
         * Implement #2 here. Finish #1 first
        if (read_ini<import_state>(module_data_dir() / "import.ini") != state) {
          fs::create_directories(module_data_dir());

          std::clog << "Footpaths Import done!" << std::endl;
          write_ini(module_data_dir() / "import.ini", state);
        }
        */

        import_successful_ = true;
      })
      ->require("NIGIRI",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_NigiriEvent;
                })
      ->require("OSM",
                [](msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSMEvent;
                })
      ->require("PPR", [](msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_PPREvent;
      });
}

}  // namespace motis::footpaths
