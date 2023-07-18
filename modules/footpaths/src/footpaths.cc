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
#include "motis/footpaths/stringmatching.h"
#include "motis/footpaths/transfer_requests.h"
#include "motis/footpaths/transfer_updates.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"

#include "osmium/io/reader.hpp"

#include "ppr/common/routing_graph.h"
#include "ppr/routing/input_pt.h"
#include "ppr/serialization/reader.h"

#include "utl/parallel_for.h"
#include "utl/verify.h"

using namespace motis::logging;
using namespace motis::module;
using namespace ppr::serialization;
using namespace ppr::routing;

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

        auto progress_tracker = utl::get_active_progress_tracker();

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

        // (profile-name, profile-info)
        progress_tracker->status("Extract Profile Information.");
        motis::ppr::read_profile_files(
            utl::to_vec(*ppr_event->profiles(),
                        [](auto const& p) { return p->path()->str(); }),
            ppr_profiles_);

        // generate (profile_name, profile position in footpath)
        // REMARK: there is no need to use the default profile
        for (auto& p : ppr_profiles_) {
          // convert walk_duration from minutes to seconds
          p.second.profile_.duration_limit_ = max_walk_duration_ * 60;

          // build profile_name to idx map in nigiri::tt
          impl_->tt_.locations_.profile_idx_.insert(
              {p.first, impl_->tt_.locations_.profile_idx_.size()});

          // build list of profile infos
          profiles_.emplace_back(p.second);
        }
        assert(impl_->tt_.locations_.profile_idx_.size() == profiles_.size());

        // Implementation of footpaths is inspired by parking

        // 1st extract all platforms from a given osm file
        progress_tracker->status("Extract Platforms from OSM.");
        std::vector<platform_info> extracted_platforms;
        auto const osm_file = osm_event->path()->str();
        {
          scoped_timer const timer{
              "transfers: extract all platforms from a given osm file."};

          LOG(info) << "Extracting platforms from " << osm_file;
          extracted_platforms = extract_osm_platforms(osm_file);
        }

        progress_tracker->status("Load PPR Routing Graph.");
        ::ppr::routing_graph rg;
        {
          scoped_timer const timer{"transfers: loading ppr routing graph."};
          read_routing_graph(rg, ppr_event->graph_path()->str());
        }

        {
          scoped_timer const timer{"transfers: preparing ppr rtrees."};
          rg.prepare_for_routing(
              edge_rtree_max_size_, area_rtree_max_size_,
              lock_rtrees_ ? rtree_options::LOCK : rtree_options::PREFETCH);
        }

        // 2nd extract all stations from the nigiri graph
        progress_tracker->status("Extract Stations from Nigiri.");
        std::vector<platform_info> stations{};
        {
          scoped_timer const timer{
              "transfers: extract stations from nigiri graph."};

          uint16_t not_in_bb = 0;
          routing_options const ro{};

          for (auto i = nigiri::location_idx_t{0U};
               i != impl_->tt_.locations_.ids_.size(); ++i) {
            if (impl_->tt_.locations_.types_[i] ==
                nigiri::location_type::kStation) {
              input_location il;
              location lo{};
              lo.set_lat(impl_->tt_.locations_.coordinates_[i].lat_);
              lo.set_lon(impl_->tt_.locations_.coordinates_[i].lng_);
              il.location_ = lo;

              if (!has_nearest_edge(rg, il, ro, false)) {
                ++not_in_bb;
                continue;
              }

              auto const name = impl_->tt_.locations_.names_[i].view();
              stations.emplace_back(std::string{name}, i,
                                    impl_->tt_.locations_.coordinates_[i]);
            }
          }
          LOG(info) << "Found " << stations.size()
                    << " stations in nigiri graph. Not in Bounding Box: "
                    << not_in_bb;
        }

        // 3rd combine platforms and stations
        progress_tracker->status("Concat. extracted platforms and stations.");
        {
          scoped_timer const timer{
              "transfers: combine single platforms and stations, build rtree."};
          extracted_platforms.insert(extracted_platforms.end(),
                                     stations.begin(), stations.end());

          LOG(info) << "Added " << stations.size()
                    << " stations to osm-extracted platforms.";

          platforms_ =
              std::make_unique<platforms>(platforms{extracted_platforms});
        }

        // 4th update osm_id and location_idx
        progress_tracker->status("Match Locations and OSM Platforms.");
        {
          scoped_timer const timer{
              "transfers: match locations and osm platforms"};
          match_locations_and_platforms();
        }

        // 5th build transfer requests
        progress_tracker->status("Generate Transfer Requests.");
        std::vector<transfer_requests> transfer_reqs;
        {
          scoped_timer const timer{"transfer: build transfer requests."};
          transfer_reqs =
              build_transfer_requests(platforms_.get(), ppr_profiles_);
        }

        // 6th get transfer requests result
        {
          scoped_timer const timer{"transfers: delete default transfers."};
          impl_->tt_.locations_.footpaths_in_.clear();
          impl_->tt_.locations_.footpaths_out_.clear();

          for (uint p_idx = 0; p_idx < ppr_profiles_.size(); ++p_idx) {
            impl_->tt_.locations_.footpaths_in_.emplace_back();
            impl_->tt_.locations_.footpaths_out_.emplace_back();

            for (uint i = 0; i < impl_->tt_.locations_.src_.size(); ++i) {
              impl_->tt_.locations_.footpaths_out_[p_idx].emplace_back(
                  nigiri::vector<nigiri::footpath>());
              impl_->tt_.locations_.footpaths_in_[p_idx].emplace_back(
                  nigiri::vector<nigiri::footpath>());
            }
          }
        }

        progress_tracker->status("Compute Transfer Routes.");
        {
          scoped_timer const timer{"transfers: update nigiri transfers"};
          precompute_nigiri_transfers(rg, impl_->tt_, ppr_profiles_,
                                      transfer_reqs);
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

void footpaths::match_locations_and_platforms() {
  // --- initialization:
  // initialize match distance range
  auto const match_distances = boost::irange(
      match_distance_min_, match_distance_max_ + match_distance_step_,
      match_distance_step_);

  // --- matching:
  u_int matched_ = 0, unmatched_ = 0;

  auto progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->reset_bounds().in_high(impl_->tt_.locations_.ids_.size());

  for (auto i = 0U; i < impl_->tt_.locations_.ids_.size(); ++i) {
    auto idx = nigiri::location_idx_t{i};
    progress_tracker->increment();
    if (impl_->tt_.locations_.types_[idx] == nigiri::location_type::kStation) {
      continue;
    }

    // match location and platform using exact name match
    auto matched = match_by_distance(idx, match_distances, exact_str_match);

    // if no exact match was found: try regex name match
    if (!matched) {
      matched =
          match_by_distance(idx, match_distances, exact_first_number_match);
    }

    if (matched) {
      ++matched_;
    } else {
      ++unmatched_;
      platforms_->platforms_.emplace_back(
          std::string(impl_->tt_.locations_.names_[idx].view()), idx,
          impl_->tt_.locations_.coordinates_[idx]);
    }
  }

  LOG(info) << "Matched " << matched_
            << " nigiri::locations to an osm-extracted platform.";
  LOG(info) << "Did not match " << unmatched_
            << " nigiri::locations to an osm-extracted platform.";
}

bool footpaths::match_by_distance(
    nigiri::location_idx_t const i,
    boost::strided_integer_range<int> match_distances,
    std::function<bool(std::string, std::string_view)> matches) {
  auto loc = impl_->tt_.locations_.coordinates_[i];
  bool matched_location{false};

  for (auto dist : match_distances) {
    for (auto* platform : platforms_->get_platforms_in_radius(loc, dist)) {
      // only match bus stops with a distance of up to a certain distance
      if (platform->is_bus_stop_ && dist > match_bus_stop_max_distance_) {
        continue;
      }

      // only match platforms with location if they have the same name
      if (!matches(platform->name_, impl_->tt_.locations_.names_[i].view())) {
        continue;
      }
      // only match platforms with a valid osm id
      if (platform->osm_id_ == -1) {
        continue;
      }

      // matched: update osm_id and osm_type of location to match platform
      // TODO (Carsten) use scoped lock here
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
}

}  // namespace motis::footpaths
