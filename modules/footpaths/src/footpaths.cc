#include "motis/footpaths/footpaths.h"

#include "cista/memory_holder.h"
#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"

#include "nigiri/timetable.h"

using namespace motis::logging;
using namespace motis::module;

namespace motis::footpaths {

struct footpaths::impl {
  nigiri::timetable tt_;
};

footpaths::footpaths() : module("Footpaths", "footpaths") {
  param(max_walk_duration_, "max_walk_duration",
        "Maximum walking time per path in minutes.");
  param(wheelchair_, "wheelchair",
        "add wheelchair profile based footpath routing");
}

footpaths::~footpaths() = default;

void footpaths::init(motis::module::registry& reg) { std::ignore = reg; }

void footpaths::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "footpaths", reg,
      [this](event_collector::dependencies_map_t const& dependencies,
             event_collector::publish_fn_t const&) {
        using import::OSMEvent;
        using import::PPREvent;

        impl_ = std::make_unique<impl>();
        impl_->tt_ = *get_shared_data<nigiri::timetable*>(
            to_res_id(global_res_id::NIGIRI_TIMETABLE));

        auto const osm_event = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const ppr_event = motis_content(PPREvent, dependencies.at("PPR"));

        uint16_t const& no_profiles = 1 + uint16_t{wheelchair_};
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

        LOG(logging::info) << "ppr profiles: ";
        for (auto const& ppr_profile : *ppr_event->profiles()) {
          LOG(info) << ppr_profile->name()->view();
        }

        std::ignore = osm_event;
        std::ignore = ppr_event;

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
