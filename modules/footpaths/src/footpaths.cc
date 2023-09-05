#include "motis/footpaths/footpaths.h"

#include <filesystem>
#include <map>
#include <regex>
#include <utility>

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/footpaths/database.h"
#include "motis/footpaths/keys.h"
#include "motis/footpaths/matching.h"
#include "motis/footpaths/platforms.h"
#include "motis/footpaths/state.h"
#include "motis/footpaths/transfer_requests.h"
#include "motis/footpaths/transfer_results.h"
#include "motis/footpaths/transfers_to_footpaths_preprocessing.h"
#include "motis/footpaths/types.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "ppr/common/routing_graph.h"
#include "ppr/serialization/reader.h"

#include "utl/parallel_for.h"
#include "utl/pipes.h"
#include "utl/zip.h"

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

  // import ppr state
  mm::named<cista::hash_t, MOTIS_NAME("ppr_graph_hash")> ppr_graph_hash_;
  mm::named<cista::hash_t, MOTIS_NAME("ppr_profiles_hash")> ppr_profiles_hash_;
};

struct footpaths::impl {
  explicit impl(n::timetable& tt,
                std::map<std::string, ppr::profile_info> const& ppr_profiles,
                std::string const& db_file, std::size_t db_max_size)
      : tt_(tt), db_{db_file, db_max_size} {
    load_ppr_profiles(ppr_profiles);

    auto const pfs = db_.get_platforms();
    old_state_.pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{pfs});
    old_state_.set_pfs_idx_ = true;
    old_state_.matches_ = db_.get_loc_to_pf_matchings();
    old_state_.transfer_requests_keys_ =
        db_.get_transfer_requests_keys(used_profiles_);
    old_state_.transfer_results_ = db_.get_transfer_results(used_profiles_);

    auto matched_pfs = platforms{};
    auto matched_nloc_keys = vector<nlocation_key_t>{};
    for (auto const& [k, pf] : old_state_.matches_) {
      matched_nloc_keys.emplace_back(k);
      matched_pfs.emplace_back(pf);
    }
    old_state_.nloc_keys_ = matched_nloc_keys;
    old_state_.matched_pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{matched_pfs});
    old_state_.set_matched_pfs_idx_ = true;
  };

  void full_import() {
    // 1st extract all platforms from a given osm file
    get_and_save_osm_platforms();

    // 2nd update osm_id and location_idx
    match_and_save_matches();

    // 3rd build transfer requests
    build_and_save_transfer_requests();

    // 4th precompute profilebased transfers
    auto rg = get_routing_ready_ppr_graph();
    route_and_save_results(rg, update_state_.transfer_requests_keys_);

    // 5th update timetable
    update_timetable(nigiri_dump_path_);
  }

  void maybe_partial_import() {
    auto const fup = get_first_update();
    auto const rt = get_routing_type(fup);

    switch (fup) {
      case first_update::kNoUpdate: break;
      case first_update::kOSM: get_and_save_osm_platforms();
      case first_update::kTimetable:
        match_and_save_matches();
        build_and_save_transfer_requests();
        break;
      case first_update::kProfiles:
        build_and_save_transfer_requests(true);
        break;
    }

    ::ppr::routing_graph rg;
    switch (rt) {
      case routing_type::kNoRouting: break;
      case routing_type::kPartialRouting:
        // do not load ppr graph if there are no routing requests
        if (update_state_.transfer_requests_keys_.empty()) {
          break;
        }
        rg = get_routing_ready_ppr_graph();
        route_and_save_results(rg, update_state_.transfer_requests_keys_);
        break;
      case routing_type::kFullRouting:
        rg = get_routing_ready_ppr_graph();
        route_and_update_results(rg, old_state_.transfer_requests_keys_);
        route_and_save_results(rg, update_state_.transfer_requests_keys_);
        break;
    }

    // update timetable
    update_timetable(nigiri_dump_path_);
  }

  std::string osm_path_;
  std::string ppr_rg_path_;
  fs::path nigiri_dump_path_;

  std::optional<import_state> old_import_state_;
  import_state new_import_state_;

private:
  // -- helper --
  void load_ppr_profiles(
      std::map<std::string, ppr::profile_info> const& ppr_profiles_by_name) {
    auto profile_names = std::vector<string>{};

    for (auto const& [pname, pinfo] : ppr_profiles_by_name) {
      profile_names.emplace_back(pname);
    }

    db_.put_profiles(profile_names);
    ppr_profile_keys_ = db_.get_profile_keys();

    for (auto& [pname, pinfo] : ppr_profiles_by_name) {
      auto pkey = ppr_profile_keys_.at(pname);
      used_profiles_.insert(pkey);

      // convert walk_duration from minutes to seconds
      ppr_profiles_.insert(std::pair<profile_key_t, ppr::profile_info>(
          pkey, ppr_profiles_by_name.at(pname)));
      ppr_profiles_.at(pkey).profile_.duration_limit_ = ::motis::MAX_WALK_TIME;

      // build profile_name to idx map in nigiri::tt
      tt_.profiles_.insert({pname, tt_.profiles_.size()});
    }
    assert(tt_.profiles_.size() == used_profiles_.size());
  }

  first_update get_first_update() {
    utl::verify(old_import_state_.has_value(), "no old import state given.");
    auto const old_import_state = old_import_state_.value();

    auto fup = first_update::kNoUpdate;
    if (old_import_state.ppr_profiles_hash_ !=
        new_import_state_.ppr_profiles_hash_) {
      fup = first_update::kProfiles;
    }

    if (old_import_state.nigiri_hash_ != new_import_state_.nigiri_hash_) {
      fup = first_update::kTimetable;
    }

    if (old_import_state.osm_path_ != new_import_state_.osm_path_) {
      fup = first_update::kOSM;
    }

    return fup;
  }

  routing_type get_routing_type(first_update const fup) {
    utl::verify(old_import_state_.has_value(), "no old import state given.");
    auto const old_import_state = old_import_state_.value();

    auto rt = routing_type::kNoRouting;
    // define routing type
    if (old_import_state.ppr_graph_hash_ != new_import_state_.ppr_graph_hash_) {
      rt = routing_type::kFullRouting;
    }

    if (fup != first_update::kNoUpdate && rt == routing_type::kNoRouting) {
      rt = routing_type::kPartialRouting;
    }

    return rt;
  }

  // -- osm platform/stop extraction --
  void get_and_save_osm_platforms() {
    progress_tracker_->status("Extract Platforms from OSM.");
    LOG(ml::info) << "Extracting platforms from " << osm_path_;
    auto osm_extracted_platforms = extract_osm_platforms(osm_path_);

    LOG(ml::info) << "Writing OSM Platforms to DB.";
    put_platforms(osm_extracted_platforms);
  }

  // -- location to osm matching --
  void match_and_save_matches() {
    progress_tracker_->status("Match Locations and OSM Platforms");
    ml::scoped_timer const timer{
        "Matching timetable locations and osm platforms."};

    auto mrs = match_locations_and_platforms(
        {tt_.locations_, old_state_, update_state_},
        {max_matching_dist_, max_bus_stop_matching_dist_});

    LOG(ml::info) << "Writing Matchings to DB.";
    put_matching_results(mrs);
  }

  // -- build transfer requests --
  void build_and_save_transfer_requests(bool const old_to_old = false) {
    progress_tracker_->status("Generating Transfer Requests.");
    ml::scoped_timer const timer{"Generating Transfer Requests."};

    auto treqs_k = generate_transfer_requests_keys(old_state_, update_state_,
                                                   ppr_profiles_, old_to_old);

    LOG(ml::info) << "Writing Transfer Requests (Keys) to DB.";
    put_transfer_requests_keys(treqs_k);
  }

  // -- build transfer results --
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

  void route_and_save_results(::ppr::routing_graph const& rg,
                              transfer_requests_keys const& treqs_k) {
    progress_tracker_->status("Precomputing Profilebased Transfers.");
    ml::scoped_timer const timer{"Precomputing Profilebased Transfers."};

    auto matches = old_state_.matches_;
    matches.insert(update_state_.matches_.begin(),
                   update_state_.matches_.end());

    auto treqs = to_transfer_requests(treqs_k, matches);
    auto trs = route_multiple_requests(treqs, rg, ppr_profiles_);
    put_transfer_results(trs);
  }

  void route_and_update_results(::ppr::routing_graph const& rg,
                                transfer_requests_keys const& treqs_k) {
    progress_tracker_->status("Updating Profilebased Transfers.");
    ml::scoped_timer const timer{"Updating Profilebased Transfers."};

    auto matches = old_state_.matches_;
    matches.insert(update_state_.matches_.begin(),
                   update_state_.matches_.end());

    auto treqs = to_transfer_requests(treqs_k, matches);
    auto trs = route_multiple_requests(treqs, rg, ppr_profiles_);
    update_transfer_results(trs);
    old_state_.transfer_results_ = db_.get_transfer_results(used_profiles_);
  }

  // -- update timetable --
  void reset_timetable() {
    for (auto prf_idx = n::profile_idx_t{0}; prf_idx < n::kMaxProfiles;
         ++prf_idx) {
      tt_.locations_.footpaths_out_[prf_idx] =
          n::vecvec<n::location_idx_t, n::footpath>{};
      tt_.locations_.footpaths_in_[prf_idx] =
          n::vecvec<n::location_idx_t, n::footpath>{};
    }
  }

  void update_timetable(fs::path const& dir) {
    progress_tracker_->status("Preprocessing Footpaths.");
    ml::scoped_timer const timer{"Updating Timetable."};

    auto key_to_name = db_.get_profile_key_to_name();

    reset_timetable();
    auto pp_fps =
        to_preprocessed_footpaths({tt_.locations_.coordinates_, tt_.profiles_,
                                   key_to_name, old_state_, update_state_});

    progress_tracker_->status("Updating Timetable.");

    // transfer footpaths from mutable_fws_multimap to timetable vecvec
    for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
         ++prf_idx) {
      for (auto loc_idx = n::location_idx_t{0U}; loc_idx < tt_.n_locations();
           ++loc_idx) {
        tt_.locations_.footpaths_out_[prf_idx].emplace_back(
            pp_fps.out_[prf_idx][loc_idx]);
      }
    }

    for (auto prf_idx = n::profile_idx_t{0U}; prf_idx < n::kMaxProfiles;
         ++prf_idx) {
      for (auto loc_idx = n::location_idx_t{0U}; loc_idx < tt_.n_locations();
           ++loc_idx) {
        tt_.locations_.footpaths_in_[prf_idx].emplace_back(
            pp_fps.in_[prf_idx][loc_idx]);
      }
    }

    tt_.write(dir);
  }

  // -- db calls --
  void put_platforms(platforms& pfs) {
    auto added_to_db = db_.put_platforms(pfs);
    auto new_pfs = utl::all(added_to_db) |
                   utl::transform([&](std::size_t i) { return pfs[i]; }) |
                   utl::vec();
    LOG(ml::info) << "Added " << added_to_db.size() << " new platforms to db.";

    LOG(ml::info) << "Building Update-State R.Tree.";
    update_state_.pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{new_pfs});
    update_state_.set_pfs_idx_ = true;
  }

  void put_matching_results(matching_results const& mrs) {
    auto added_to_db = db_.put_matching_results(mrs);
    auto new_mrs = utl::all(added_to_db) |
                   utl::transform([&](std::size_t i) { return mrs[i]; }) |
                   utl::vec();

    auto matched_pfs = platforms{};
    for (auto const& mr : new_mrs) {
      update_state_.matches_.insert(
          std::pair<nlocation_key_t, platform>(to_key(mr.nloc_pos_), mr.pf_));
      update_state_.nloc_keys_.emplace_back(to_key(mr.nloc_pos_));
      matched_pfs.emplace_back(mr.pf_);
    }

    update_state_.matched_pfs_idx_ =
        std::make_unique<platforms_index>(platforms_index{matched_pfs});
    update_state_.set_matched_pfs_idx_ = true;
  }

  /**
   * save new or update old transfer requests
   */
  void put_transfer_requests_keys(transfer_requests_keys const treqs_k) {
    auto updated_in_db = db_.update_transfer_requests_keys(treqs_k);
    auto added_to_db = db_.put_transfer_requests_keys(treqs_k);
    auto updated_treqs_k =
        utl::all(updated_in_db) |
        utl::transform([&](std::size_t i) { return treqs_k[i]; }) | utl::vec();
    auto new_treqs_k =
        utl::all(added_to_db) |
        utl::transform([&](std::size_t i) { return treqs_k[i]; }) | utl::vec();

    update_state_.transfer_requests_keys_ = new_treqs_k;
  }

  void put_transfer_results(transfer_results const& trs) {
    auto added_to_db = db_.put_transfer_results(trs);
    assert(trs.size() == added_to_db.size());
    update_state_.transfer_results_ = trs;
  }

  void update_transfer_results(transfer_results const& trs) {
    auto updated_in_db = db_.update_transfer_results(trs);
    assert(trs.size() == updated_in_db.size());
  }

  n::timetable& tt_;
  database db_;

  hash_map<nlocation_key_t, n::location_idx_t> location_key_to_idx_;

  hash_map<string, profile_key_t> ppr_profile_keys_;
  hash_map<profile_key_t, ppr::profile_info> ppr_profiles_;
  set<profile_key_t> used_profiles_;

  state old_state_;  // state before init/import
  state update_state_;  // update state with new platforms/new matches

  // initialize matching limits
  double max_matching_dist_{20};
  double max_bus_stop_matching_dist_{120};

  // initialize progress tracker (ptr)
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
        using import::FileEvent;
        using import::NigiriEvent;
        using import::PPREvent;

        auto const dir = get_data_directory() / "footpaths";
        auto const nigiri =
            motis_content(NigiriEvent, dependencies.at("NIGIRI"));
        auto const files = motis_content(FileEvent, dependencies.at("FILES"));
        auto const ppr = motis_content(PPREvent, dependencies.at("PPR"));

        // extract osm path from files
        std::string osm_path;
        for (auto const& p : *files->paths()) {
          if (p->tag()->str() == "osm") {
            osm_path = p->path()->str();
            break;
          }
        }
        utl::verify(!osm_path.empty(), "no osm file given.");

        auto const new_import_state = import_state{
            nigiri->hash(), osm_path, ppr->graph_hash(), ppr->profiles_hash()};

        auto ppr_profiles = std::map<std::string, ppr::profile_info>{};
        ::motis::ppr::read_profile_files(
            utl::to_vec(*ppr->profiles(),
                        [](auto const& p) { return p->path()->str(); }),
            ppr_profiles);

        fs::create_directories(dir);
        impl_ = std::make_unique<impl>(
            *get_shared_data<n::timetable*>(
                to_res_id(mm::global_res_id::NIGIRI_TIMETABLE)),
            ppr_profiles, db_file(), db_max_size_);

        impl_->new_import_state_ = new_import_state;
        impl_->osm_path_ = osm_path;
        impl_->ppr_rg_path_ = ppr->graph_path()->str();
        impl_->nigiri_dump_path_ =
            get_data_directory() / "nigiri" / fmt::to_string(nigiri->hash());

        {
          ml::scoped_timer const timer{"Footpath Import"};
          if (!fs::exists(dir / "import.ini")) {
            LOG(ml::info) << "Footpaths: Full Import.";
            impl_->full_import();
            import_successful_ = true;
          } else {
            impl_->old_import_state_ =
                mm::read_ini<import_state>(dir / "import.ini");
            if (impl_->old_import_state_.value() != impl_->new_import_state_) {
              LOG(ml::info) << "Footpaths: Maybe Partial Import.";
              impl_->maybe_partial_import();
            }
            import_successful_ = true;
          }
        }

        mm::write_ini(dir / "import.ini", new_import_state);
      })
      ->require("FILES",
                [](mm::msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_FileEvent;
                })
      ->require("NIGIRI",
                [](mm::msg_ptr const& msg) {
                  return msg->get()->content_type() == MsgContent_NigiriEvent;
                })
      ->require("PPR", [](mm::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_PPREvent;
      });
}

}  // namespace motis::footpaths
