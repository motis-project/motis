#include "motis/footpaths/footpaths.h"

#include <iostream>

#include "utl/verify.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"

#include "nigiri/timetable.h"

using namespace motis::logging;
using namespace motis::module;

namespace motis::footpaths {

footpaths::footpaths() : module("Footpaths", "footpaths") {
  // to add module parameters:
  // - add field to module struct (footpaths.h), e.g. int foo_;
  // - declare the parameter here, e.g.:
  //   param(foo_, "foo", "description");
  // - use command line argument "--footpaths.foo value"
  //   or in config.ini:
  //   [footpaths]
  //   foo=value
}

footpaths::~footpaths() = default;

void footpaths::init(motis::module::registry& reg) {}

void footpaths::import(motis::module::import_dispatcher& reg) {
  std::make_shared<event_collector>(
      get_data_directory().generic_string(), "footpaths", reg,
      [this](event_collector::dependencies_map_t const& dependencies,
             event_collector::publish_fn_t const&) {
        using import::OSMEvent;
        using import::PPREvent;

        auto& tt = *get_shared_data<nigiri::timetable*>(
            to_res_id(global_res_id::NIGIRI_TIMETABLE));
        auto const osm_ev = motis_content(OSMEvent, dependencies.at("OSM"));
        auto const ppr_ev = motis_content(PPREvent, dependencies.at("PPR"));

        // log output is written to ${data}/log/${module name}.txt

        auto const osm_file = osm_ev->path()->str();
        LOG(info) << "loading " << osm_file << "...";

        LOG(info) << "ppr profiles:";
        for (auto const& p : *ppr_ev->profiles()) {
          LOG(info) << p->name()->view();
        }

        for (auto i = nigiri::location_idx_t{0U};
             i != tt.locations_.ids_.size(); ++i) {
          auto const loc_type = tt.locations_.types_[i];
          if (loc_type == nigiri::location_type::kStation) {
            auto const coords = tt.locations_.coordinates_[i];
            std::clog << "station " << tt.locations_.names_[i].view()
                      << ": lat=" << coords.lat_ << ", lng=" << coords.lng_
                      << "\n";
          } else if (loc_type == nigiri::location_type::kTrack) {
            auto const parent_idx = tt.locations_.parents_[i];
            utl::verify(parent_idx != nigiri::location_idx_t::invalid(),
                        "track without parent");
            std::clog << "track " << tt.locations_.names_[i].view()
                      << " at station "
                      << tt.locations_.names_[parent_idx].view() << "\n";
          }
        }

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
