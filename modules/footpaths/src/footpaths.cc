#include "motis/footpaths/footpaths.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <regex>

#include "boost/range/irange.hpp"

#include "cista/containers/vector.h"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/footpaths/database.h"
#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"
#include "motis/footpaths/transfer_requests.h"
#include "motis/footpaths/transfer_updates.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"

#include "ppr/common/routing_graph.h"
#include "ppr/routing/input_pt.h"
#include "ppr/serialization/reader.h"

#include "utl/parallel_for.h"
#include "utl/verify.h"

namespace cr = cista::raw;
namespace fs = std::filesystem;
namespace ml = motis::logging;
namespace mm = motis::module;
namespace n = nigiri;
namespace pr = ppr::routing;
namespace ps = ppr::serialization;

namespace motis::footpaths {

// load already known/stored data
struct import_state {
  CISTA_COMPARABLE();
  // import nigiri state
  mm::named<cista::hash_t, MOTIS_NAME("nigiri_hash")> nigiri_hash_;

  // import osm state
  mm::named<std::string, MOTIS_NAME("osm_path")> osm_path_;
  mm::named<cista::hash_t, MOTIS_NAME("osm_hash")> osm_hash_;
  mm::named<std::size_t, MOTIS_NAME("osm_size")> osm_size_;

  // import ppr state
  mm::named<std::string, MOTIS_NAME("ppr_graph_path")> ppr_graph_path_;
  mm::named<cista::hash_t, MOTIS_NAME("ppr_graph_hash")> ppr_graph_hash_;
  mm::named<std::size_t, MOTIS_NAME("ppr_graph_size")> ppr_graph_size_;
  mm::named<cista::hash_t, MOTIS_NAME("ppr_profiles_hash")> ppr_profiles_hash_;
  mm::named<int, MOTIS_NAME("max_walk_duration")> max_walk_duration_;
};

struct footpaths::impl {
  explicit impl(nigiri::timetable& tt, std::string const& db_file,
                std::size_t db_max_size)
      : tt_(tt), db_{db_file, db_max_size} {};

  void match_locations_and_platforms() {
    // --- initialization: initialize match distance range
    auto const dists = boost::irange(match_distance_min_,
                                     match_distance_max_ + match_distance_step_,
                                     match_distance_step_);

    // --- matching:
    unsigned int matched_ = 0, unmatched_ = 0;

    auto progress_tracker = utl::get_active_progress_tracker();
    progress_tracker->reset_bounds().in_high(tt_.locations_.ids_.size());

    for (auto i = 0U; i < tt_.locations_.ids_.size(); ++i) {
      progress_tracker->increment();

      auto nloc = tt_.locations_.get(n::location_idx_t{i});
      if (nloc.type_ == n::location_type::kStation) {
        continue;
      }

      // match location and platform using exact name match
      auto [has_match, match_res] = match_by_name(nloc, pfs_idx_.get(), dists,
                                                  match_bus_stop_max_distance_);

      if (has_match) {
        ++matched_;
        loc_osm_ids_[match_res.loc_idx_] = match_res.pf_->info_.osm_id_;
        loc_osm_types_[match_res.loc_idx_] = match_res.pf_->info_.osm_type_;

        if (match_res.pf_->info_.idx_ == n::location_idx_t::invalid()) {
          // TODO (Carsten) allow multiple loc_idxs per platform;
          match_res.pf_->info_.idx_ = match_res.loc_idx_;
        }
        continue;
      } else {
        ++unmatched_;
        pfs_idx_->platforms_.emplace_back(platform{
            0, nloc.pos_,
            platform_info{nloc.name_, -1, nloc.l_, osm_type::kNode, false}});
      }
    }

    LOG(ml::info) << "Matched " << matched_
                  << " nigiri::locations to an osm-extracted platform.";
    LOG(ml::info) << "Did not match " << unmatched_
                  << " nigiri::locations to an osm-extracted platform.";
  }

  nigiri::timetable& tt_;

private:
  database db_;

  int match_distance_min_{0};
  int match_distance_max_{400};
  int match_distance_step_{40};
  int match_bus_stop_max_distance_{120};

  cr::vector_map<n::location_idx_t, std::int64_t> loc_osm_ids_;
  cr::vector_map<n::location_idx_t, osm_type> loc_osm_types_;

  std::unique_ptr<platforms_index> pfs_idx_;
};

footpaths::footpaths() : module("Footpaths", "footpaths") {
  param(max_walk_duration_, "max_walk_duration",
        "Maximum walking time per path in minutes.");
}

footpaths::~footpaths() = default;

fs::path footpaths::module_data_dir() const {
  return get_data_directory() / "footpaths";
}

std::string footpaths::db_file() const {
  return (module_data_dir() / "footpaths.db").generic_string();
}

void footpaths::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "footpaths", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::NigiriEvent;
        using import::OSMEvent;
        using import::PPREvent;

        auto const dir = get_data_directory() / "footpaths";
        auto const nigiri_event =
            motis_content(NigiriEvent, dependencies.at("NIGIRI"));
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
        auto const& no_profiles = ppr_event->profiles()->size();
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
        ::motis::ppr::read_profile_files(
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

        // 1st extract all platforms from a given osm file
        progress_tracker->status("Extract Platforms from OSM.");
        std::vector<platform> extracted_platforms;
        auto const osm_file = osm_event->path()->str();
        {
          ml::scoped_timer const timer{
              "transfers: extract all platforms from a given osm file."};

          LOG(ml::info) << "Extracting platforms from " << osm_file;
          extracted_platforms = extract_osm_platforms(osm_file);
        }

        progress_tracker->status("Load PPR Routing Graph.");
        ::ppr::routing_graph rg;
        {
          ml::scoped_timer const timer{"transfers: loading ppr routing graph."};
          ps::read_routing_graph(rg, ppr_event->graph_path()->str());
        }

        {
          ml::scoped_timer const timer{"transfers: preparing ppr rtrees."};
          rg.prepare_for_routing(edge_rtree_max_size_, area_rtree_max_size_,
                                 lock_rtrees_ ? ::ppr::rtree_options::LOCK
                                              : ::ppr::rtree_options::PREFETCH);
        }

        // 2nd extract all stations from the nigiri graph
        progress_tracker->status("Extract Stations from Nigiri.");
        std::vector<platform> stations{};
        {
          ml::scoped_timer const timer{
              "transfers: extract stations from nigiri graph."};

          uint16_t not_in_bb = 0;
          pr::routing_options const ro{};

          for (auto i = nigiri::location_idx_t{0U};
               i != impl_->tt_.locations_.ids_.size(); ++i) {
            if (impl_->tt_.locations_.types_[i] ==
                nigiri::location_type::kStation) {
              auto const il =
                  ::ppr::routing::make_input_location(::ppr::make_location(
                      impl_->tt_.locations_.coordinates_[i].lng_,
                      impl_->tt_.locations_.coordinates_[i].lat_));

              if (!has_nearest_edge(rg, il, ro, false)) {
                ++not_in_bb;
                continue;
              }

              stations.emplace_back(
                  platform{0, impl_->tt_.locations_.coordinates_[i],
                           platform_info{impl_->tt_.locations_.names_[i].view(),
                                         -1, i, osm_type::kNode, false}});
            }
          }
          LOG(ml::info) << "Found " << stations.size()
                        << " stations in nigiri graph. Not in Bounding Box: "
                        << not_in_bb;
        }

        // 3rd combine platforms and stations
        progress_tracker->status("Concat. extracted platforms and stations.");
        {
          ml::scoped_timer const timer{
              "transfers: combine single platforms and stations, build rtree."};
          extracted_platforms.insert(extracted_platforms.end(),
                                     stations.begin(), stations.end());

          LOG(ml::info) << "Added " << stations.size()
                        << " stations to osm-extracted platforms.";

          // platforms_ =
          //     std::make_unique<platforms>(platforms{extracted_platforms});
        }

        // 4th update osm_id and location_idx
        progress_tracker->status("Match Locations and OSM Platforms.");
        {
          ml::scoped_timer const timer{
              "transfers: match locations and osm platforms"};
          // match_locations_and_platforms();
        }

        // 5th build transfer requests
        progress_tracker->status("Generate Transfer Requests.");
        std::vector<transfer_requests> const transfer_reqs;
        {
          ml::scoped_timer const timer{"transfer: build transfer requests."};
          // transfer_reqs =
          //    build_transfer_requests(platforms_.get(), ppr_profiles_);
        }

        // 6th get transfer requests result
        {
          ml::scoped_timer const timer{"transfers: delete default transfers."};
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
          ml::scoped_timer const timer{"transfers: update nigiri transfers"};
          precompute_nigiri_transfers(rg, impl_->tt_, ppr_profiles_,
                                      transfer_reqs);
        }

        LOG(ml::info) << "Footpath Import done!";
        mm::write_ini(dir / "import.ini", state);

        import_successful_ = true;
      })
      ->require("NIGIRI",
                [](mm::msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_NigiriEvent;
                })
      ->require("OSM",
                [](mm::msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_OSMEvent;
                })
      ->require("PPR", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_PPREvent;
      });
}

void footpaths::init(motis::module::registry& reg) {
  std::ignore = reg;

  try {
    impl_ = std::make_unique<impl>(
        *get_shared_data<nigiri::timetable*>(
            to_res_id(mm::global_res_id::NIGIRI_TIMETABLE)),
        db_file(), db_max_size_);
  } catch (std::exception const& e) {
    LOG(ml::warn) << "footpaths module not initialized: " << e.what();
  }
}

}  // namespace motis::footpaths
