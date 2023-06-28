#include "motis/footpaths/footpaths.h"

#include <filesystem>
#include <iostream>
#include <regex>

#include "boost/range/adaptors.hpp"
#include "boost/range/irange.hpp"

#include "cista/memory_holder.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfer_requests.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"

#include "utl/parallel_for.h"
#include "utl/verify.h"

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
          // convert walk_duration from minutes to seconds
          p.second.profile_.duration_limit_ = max_walk_duration_ * 60;
          // build profile to idx map
          ppr_profile_pos_.insert({p.first, ppr_profile_pos_.size()});

          // build list of profile infos
          profiles_.emplace_back(p.second);
        }

        // Implementation of footpaths is inspired by parking

        // 1st extract all platforms from a given osm file
        std::vector<platform_info> extracted_platforms;
        {
          scoped_timer const timer{
              "transfers: extract all platforms from a given osm file."};
          auto const osm_file = osm_event->path()->str();

          LOG(info) << "Extracting platforms from " << osm_file;
          extracted_platforms = extract_osm_platforms(osm_file);
        }

        // 2nd extract all stations from the nigiri graph
        std::vector<platform_info> stations{};
        {
          scoped_timer const timer{
              "transfers: extract stations from nigiri graph."};

          for (auto i = nigiri::location_idx_t{0U};
               i != impl_->tt_.locations_.ids_.size(); ++i) {
            if (impl_->tt_.locations_.types_[i] ==
                nigiri::location_type::kStation) {
              auto const name = impl_->tt_.locations_.names_[i].view();
              stations.emplace_back(static_cast<std::string>(name), i,
                                    impl_->tt_.locations_.coordinates_[i]);
            }
          }
        }

        // 3rd combine platforms and stations
        {
          scoped_timer const timer{
              "transfers: combine single platforms and stations, build rtree."};
          extracted_platforms.insert(extracted_platforms.end(),
                                     stations.begin(), stations.end());
          platforms_ =
              std::make_unique<platforms>(platforms{extracted_platforms});
        }

        // 4th update osm_id and location_idx
        {
          scoped_timer const timer{
              "transfers: match locations and osm platforms"};
          match_locations_and_platforms();
        }

        // 5th build transfer requests
        {
          scoped_timer const timer{"transfer: build transfer requests."};
          auto transfer_reqs = build_transfer_requests(
              platforms_.get(), profiles_, max_walk_duration_);
        }

        // 6th get transfer requests result
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

void footpaths::match_locations_and_platforms() {
  // --- initialization:
  // initialize match distance range
  auto const match_distance = boost::irange(
      match_distance_min_, match_distance_max_, match_distance_step_);

  // initialize str_a == str_b lambda function
  auto exact_name_match = [](std::string str_a, std::string_view str_b) {
    return str_a == str_b;
  };

  // initialize number_a == number_b regex number match lambda
  auto exact_first_number_match = [](std::string str_a,
                                     std::string_view str_b) {
    str_a = std::regex_replace(str_a, std::regex("[^0-9]+"), std::string("$1"));
    str_b = std::regex_replace(std::string{str_b}, std::regex("[^0-9]+"),
                               std::string("$1"));
    if (str_b.length() == 0) {
      return false;
    }
    return str_a == str_b;
  };

  // initialize match location and platforms using platforms near location
  auto match_by_distance = [this, match_distance](
                               nigiri::location_idx_t const i, auto& matches) {
    auto loc = impl_->tt_.locations_.coordinates_[i];
    bool matched_location{false};

    for (auto dist : match_distance) {
      for (auto* platform :
           platforms_.get()->get_platforms_in_radius(loc, dist)) {
        // only match bus stops with a distance of up to a certain distance
        if (platform->is_bus_stop_ && dist > match_bus_stop_max_distance_) {
          continue;
        }

        // TODO (Carsten) Bus-Stops are matched within a
        // 120m radius;
        // only match platforms with location if they have the same name
        if (matches(platform->name_,
                    this->impl_->tt_.locations_.names_[i].view())) {
          continue;
        }
        // only match platforms with a valid osm id
        if (platform->osm_id_ == -1) {
          continue;
        }

        // matched: update osm_id and osm_type of location to match platform
        impl_->tt_.locations_.osm_ids_[i] =
            nigiri::osm_node_id_t{platform->osm_id_};
        impl_->tt_.locations_.osm_types_[i] = platform->osm_type_;
        if (platform->idx_ == nigiri::location_idx_t::invalid()) {
          platform->idx_ = i;
        }

        matched_location = true;
        break;
      }

      // location unmatched: increase match distance
      if (!matched_location) {
        continue;
      }

      // matched location to platform. GoTo next platform
      break;
    }
    return matched_location;
  };

  // --- matching:
  auto locations =
      boost::irange(0, static_cast<int>(impl_->tt_.locations_.ids_.size()), 1);

  utl::parallel_for(locations, [&](auto i) {
    auto location_idx = nigiri::location_idx_t{i};
    if (impl_->tt_.locations_.types_[location_idx] !=
        nigiri::location_type::kTrack) {
      return;
    }

    // match location and platform using exact name match
    auto matched = match_by_distance(location_idx, exact_name_match);

    // if no exact match was found: try regex name match
    if (!matched) {
      match_by_distance(location_idx, exact_first_number_match);
    }
  });

  return;
}

}  // namespace motis::footpaths
