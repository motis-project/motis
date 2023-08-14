#include "motis/footpaths/footpaths.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <regex>

#include "boost/range/irange.hpp"

#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/footpaths/database.h"
#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"
#include "motis/footpaths/state.h"
#include "motis/footpaths/transfer_requests.h"
#include "motis/footpaths/transfer_results.h"
#include "motis/footpaths/transfers.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "ppr/common/routing_graph.h"
#include "ppr/routing/input_pt.h"
#include "ppr/serialization/reader.h"

#include "utl/parallel_for.h"
#include "utl/pipes.h"

namespace fs = std::filesystem;
namespace ml = motis::logging;
namespace mm = motis::module;
namespace n = nigiri;
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
};

struct footpaths::impl {
  explicit impl(n::timetable& tt, std::string const& db_file,
                std::size_t db_max_size)
      : tt_(tt), db_{db_file, db_max_size} {
    auto pfs = db_.get_platforms();
    old_state_.pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{pfs});
    old_state_.matches_ = db_.get_loc_to_pf_matchings();
    old_state_.transfer_results_ = db_.get_transfer_results();

    auto matched_pfs = platforms{};
    auto matched_nloc_keys = std::vector<string>{};
    for (auto const& [k, pf] : old_state_.matches_) {
      matched_nloc_keys.emplace_back(k);
      matched_pfs.emplace_back(pf);
    }
    old_state_.nloc_keys = matched_nloc_keys;
    old_state_.matched_pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{matched_pfs});
  };

  void full_import() {
    // 1st extract all platforms from a given osm file
    get_and_save_osm_platforms();

    // 2nd update osm_id and location_idx
    match_and_save_matches();

    // 3rd build transfer requests
    auto const treqs = get_new_trs();

    // 4th precompute profilebased transfers
    auto rg = get_routing_ready_ppr_graph();
    route_and_save_result(rg, treqs);

    // 5th update timetable
    update_timetable();
  }

  void maybe_partial_import(import_state const& old_state,
                            import_state const& new_state) {
    if (old_state.osm_hash_ != new_state.osm_hash_) {
      // 1st extract all platforms from a given osm file
      get_and_save_osm_platforms();

      // 2nd update osm_id and location_idx
      match_and_save_matches();

      // 3rd build transfer requests
      auto const treqs = get_new_trs();

      // 4th precompute profilebased transfers
      auto rg = get_routing_ready_ppr_graph();
      route_and_save_result(rg, treqs);
    }

    // update timetable
    update_timetable();
  }

  void load_ppr_profiles() {
    for (auto& [pname, pinfo] : ppr_profiles_) {
      // convert walk_duration from minutes to seconds
      pinfo.profile_.duration_limit_ = max_walk_duration_ * 60;

      // build profile_name to idx map in nigiri::tt
      tt_.profiles_.insert({pname, tt_.profiles_.size()});

      // build list of profile infos
      profiles_.emplace_back(pinfo);
    }
    assert(tt_.profiles_.size() == profiles_.size());
  }

  n::timetable& tt_;
  database db_;

  state old_state_;  // state before init/import
  state update_state_;  // update state with new platforms/new matches

  std::map<std::string, ppr::profile_info> ppr_profiles_;

  std::string osm_path_;
  std::string ppr_rg_path_;

  // initialize footpaths limits
  int max_walk_duration_{10};

private:
  void get_and_save_osm_platforms() {
    progress_tracker_->status("Extract Platforms from OSM.");
    LOG(ml::info) << "Extracting platforms from " << osm_path_;
    auto osm_extracted_platforms = extract_osm_platforms(osm_path_);

    LOG(ml::info) << "Writing OSM Platforms to DB.";
    put_platforms(osm_extracted_platforms);
  }

  void match_and_save_matches() {
    progress_tracker_->status("Match Locations and OSM Platforms");
    ml::scoped_timer const timer{
        "Matching timetable locations and osm platforms."};

    LOG(ml::info) << "Matching Locations to OSM Platforms.";
    auto mrs = match_locations_and_platforms();

    LOG(ml::info) << "Writing Matchings to DB.";
    put_matching_results(mrs);
  }

  transfer_requests get_new_trs() {
    progress_tracker_->status("Generate Transfer Requests.");
    ml::scoped_timer const timer{"transfer: build transfer requests."};
    return new_all_reachable_pairs_requests();
  }

  void route_and_save_result(::ppr::routing_graph const& rg,
                             transfer_requests const& treqs) {
    progress_tracker_->status("(Pre)Computing Profilebased Transfers.");
    ml::scoped_timer const timer{"(Pre)Computing Profilebased Transfers."};
    auto trs = route_multiple_requests(treqs, rg, ppr_profiles_);
    put_transfer_results(trs);
  }

  ::ppr::routing_graph get_routing_ready_ppr_graph() {
    ::ppr::routing_graph result;
    progress_tracker_->status("Loading PPR Routing Graph.");
    ml::scoped_timer const timer{"Loading PPR Routing Graph."};
    ps::read_routing_graph(result, ppr_rg_path_);
    result.prepare_for_routing(edge_rtree_max_size_, area_rtree_max_size_,
                               lock_rtree_ ? ::ppr::rtree_options::LOCK
                                           : ::ppr::rtree_options::PREFETCH);
    return result;
  }

  platforms extract_tt_platforms() {
    auto result = platforms{};

    for (auto i = n::location_idx_t{0U}; i < tt_.locations_.ids_.size(); ++i) {
      if (tt_.locations_.types_[i] == n::location_type::kStation) {
        result.emplace_back(
            platform{0, tt_.locations_.coordinates_[i],
                     platform_info{tt_.locations_.names_[i].view(), -1, i,
                                   osm_type::kNode, false}});
      }
    }

    return result;
  }

  matching_results match_locations_and_platforms() const {
    // --- initialization: initialize match distance range
    [[maybe_unused]] auto const dists = boost::irange(
        match_distance_min_, match_distance_max_ + match_distance_step_,
        match_distance_step_);

    // --- matching:
    unsigned int matched_ = 0;
    auto matches = matching_results{};

    auto progress_tracker = utl::get_active_progress_tracker();
    progress_tracker->reset_bounds().in_high(tt_.locations_.ids_.size());

    for (auto i = 0U; i < tt_.locations_.ids_.size(); ++i) {
      progress_tracker->increment();

      auto nloc = tt_.locations_.get(n::location_idx_t{i});
      /*if (nloc.type_ == n::location_type::kStation) {
        continue;
      }*/

      // match location and platform using exact name match
      auto [has_match_up, match_res] = match_by_distance(
          nloc, old_state_, update_state_, 20, match_bus_stop_max_distance_);

      if (!has_match_up) {
        continue;
      }

      ++matched_;
      matches.emplace_back(match_res);
    }

    LOG(ml::info) << "Matched " << matched_
                  << " nigiri::locations to an osm-extracted platform.";
    return matches;
  }

  transfer_requests new_all_reachable_pairs_requests() const {
    return generate_new_all_reachable_pairs_requests(old_state_, update_state_,
                                                     ppr_profiles_);
  }

  void reset_timetable() {
    for (auto prf_idx = n::profile_idx_t{0}; prf_idx < n::kMaxProfiles;
         ++prf_idx) {
      tt_.locations_.footpaths_out_[prf_idx] = {};
      tt_.locations_.footpaths_in_[prf_idx] = {};

      for (auto i = 0; i < tt_.locations_.ids_.size(); ++i) {
        tt_.locations_.footpaths_out_[prf_idx].emplace_back(
            n::vector<n::footpath>());
        tt_.locations_.footpaths_in_[prf_idx].emplace_back(
            n::vector<n::footpath>());
      }
    }
  }

  void update_timetable() {
    progress_tracker_->status("Updating Timetable.");
    ml::scoped_timer const timer{"Updating Timetable."};

    reset_timetable();
    unsigned int ctr_start = 0;
    unsigned int ctr_end = 0;

    progress_tracker_->in_high(old_state_.transfer_results_.size() +
                               update_state_.transfer_results_.size());

    auto const& single_update = [&](transfer_result const& tr) {
      ++ctr_start;
      progress_tracker_->increment();
      if (tt_.profiles_.count(tr.profile_) == 0 ||
          tt_.locations_.location_key_to_idx_.count(tr.from_nloc_key) == 0 ||
          tt_.locations_.location_key_to_idx_.count(tr.to_nloc_key) == 0) {
        return;
      }

      auto const prf_idx = tt_.profiles_.at(tr.profile_);
      auto const from_idx =
          tt_.locations_.location_key_to_idx_.at(tr.from_nloc_key);
      auto const to_idx =
          tt_.locations_.location_key_to_idx_.at(tr.to_nloc_key);

      tt_.locations_.footpaths_out_[prf_idx][from_idx].push_back(
          n::footpath{to_idx, tr.info_.duration_});
      tt_.locations_.footpaths_in_[prf_idx][to_idx].push_back(
          n::footpath{from_idx, tr.info_.duration_});

      ++ctr_end;
    };

    for (auto const& tr : old_state_.transfer_results_) {
      single_update(tr);
    }

    for (auto const& tr : update_state_.transfer_results_) {
      single_update(tr);
    }

    LOG(ml::info) << "Added " << ctr_end << " of " << ctr_start
                  << " transfers.";
  }

  // -- db calls --
  void put_platforms(platforms& pfs) {
    auto added_to_db = db_.put_platforms(pfs);
    auto new_pfs = utl::all(added_to_db) |
                   utl::transform([&](std::size_t i) { return pfs[i]; }) |
                   utl::vec();
    LOG(ml::info) << "Added " << added_to_db.size() << " new platforms to db.";

    assert(new_pfs.size() == added_to_db.size());

    LOG(ml::info) << "Building Update-State R.Tree.";
    update_state_.pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{new_pfs});
  }

  std::vector<std::size_t> put_matching_results(matching_results const& mrs) {
    auto added_to_db = db_.put_matching_results(mrs);
    auto new_mrs = utl::all(added_to_db) |
                   utl::transform([&](std::size_t i) { return mrs[i]; }) |
                   utl::vec();
    LOG(ml::info) << "Added " << added_to_db.size()
                  << " new matching results to db.";

    assert(new_mrs.size() == added_to_db.size());

    auto matched_pfs = platforms{};
    for (auto const& mr : new_mrs) {
      update_state_.matches_.insert(
          std::pair<std::string, platform>(to_key(mr.nloc_pos_), mr.pf_));
      update_state_.nloc_keys.emplace_back(to_key(mr.nloc_pos_));
      matched_pfs.emplace_back(mr.pf_);
    }
    update_state_.matched_pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{matched_pfs});

    return added_to_db;
  }

  void put_transfer_results(transfer_results const& trs) {
    auto added_to_db = db_.put_transfer_results(trs);
    assert(trs.size() == added_to_db.size());
    LOG(ml::info) << "Added " << added_to_db.size() << " of " << trs.size()
                  << " new transfers to db.";

    update_state_.transfer_results_ = trs;
  }

  std::vector<ppr::profile_info> profiles_;

  // initialize matching limits
  int match_distance_min_{0};
  int match_distance_max_{400};
  int match_distance_step_{40};
  int match_bus_stop_max_distance_{120};

  utl::progress_tracker_ptr progress_tracker_{
      utl::get_active_progress_tracker()};
};  // namespace motis::footpaths

footpaths::footpaths() : module("Footpaths", "footpaths") {}

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
        auto const new_state =
            import_state{nigiri_event->hash(),
                         data_path(osm_event->path()->str()),
                         osm_event->hash(),
                         osm_event->size(),
                         data_path(ppr_event->graph_path()->str()),
                         ppr_event->graph_hash(),
                         ppr_event->graph_size(),
                         ppr_event->profiles_hash()};

        fs::create_directories(dir);
        impl_ = std::make_unique<impl>(
            *get_shared_data<n::timetable*>(
                to_res_id(mm::global_res_id::NIGIRI_TIMETABLE)),
            db_file(), db_max_size_);

        auto progress_tracker = utl::get_active_progress_tracker();

        {
          ml::scoped_timer const timer{"Reading PPR Profile Information"};
          progress_tracker->status("Reading PPR Profiles Information.");

          ::motis::ppr::read_profile_files(
              utl::to_vec(*ppr_event->profiles(),
                          [](auto const& p) { return p->path()->str(); }),
              impl_->ppr_profiles_);
          impl_->load_ppr_profiles();
        }

        impl_->osm_path_ = osm_event->path()->str();
        impl_->ppr_rg_path_ = ppr_event->graph_path()->str();

        if (!fs::exists(dir / "import.ini")) {
          LOG(ml::info) << "Footpaths: Full Import.";
          impl_->full_import();
        } else {
          LOG(ml::info) << "Footpaths: Maybe Partial Import.";
          auto old_state = mm::read_ini<import_state>(dir / "import.ini");
          impl_->maybe_partial_import(old_state, new_state);
        }

        LOG(ml::info) << "Footpath Import done!";
        mm::write_ini(dir / "import.ini", new_state);

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

void footpaths::init(motis::module::registry& reg) { std::ignore = reg; }

}  // namespace motis::footpaths
