#include "motis/transfers/transfers.h"

#include <map>
#include <utility>

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "motis/transfers/matching/by_distance.h"
#include "motis/transfers/platform/extract.h"
#include "motis/transfers/storage/storage.h"
#include "motis/transfers/transfer/transfer_request.h"
#include "motis/transfers/transfer/transfer_result.h"
#include "motis/transfers/types.h"

#include "motis/ppr/profiles.h"

#include "nigiri/timetable.h"
#include "nigiri/types.h"

#include "ppr/common/routing_graph.h"
#include "ppr/serialization/reader.h"

namespace fs = std::filesystem;
namespace ml = motis::logging;
namespace mm = motis::module;
namespace n = ::nigiri;
namespace ps = ::ppr::serialization;

namespace motis::transfers {

enum class first_update { kNoUpdate, kProfiles, kTimetable, kOSM };
enum class routing_type { kNoRouting, kPartialRouting, kFullRouting };

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

struct transfers::impl {
  explicit impl(n::timetable& tt,
                std::map<std::string, ppr::profile_info> const& ppr_profiles,
                fs::path const& db_file_path, std::size_t db_max_size)
      : storage_{db_file_path, db_max_size, tt} {
    load_ppr_profiles(ppr_profiles);
    storage_.initialize();
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
    route_and_save_results(rg, storage_.get_transfer_requests_keys(
                                   data_request_type::kPartialUpdate));

    // 5th update timetable
    storage_.update_tt(nigiri_dump_path_);
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
        if (storage_.has_transfer_requests_keys(
                data_request_type::kPartialUpdate)) {
          break;
        }
        rg = get_routing_ready_ppr_graph();
        // route "new" transfer requests
        route_and_save_results(rg, storage_.get_transfer_requests_keys(
                                       data_request_type::kPartialUpdate));
        break;
      case routing_type::kFullRouting:
        rg = get_routing_ready_ppr_graph();
        // reroute "old" transfer requests
        route_and_save_results(rg, storage_.get_transfer_requests_keys(
                                       data_request_type::kPartialOld));
        // route "new" transfer requests
        route_and_save_results(rg, storage_.get_transfer_requests_keys(
                                       data_request_type::kPartialUpdate));
        break;
    }

    // update timetable
    storage_.update_tt(nigiri_dump_path_);
  }

  fs::path osm_path_;
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

    storage_.add_new_profiles(profile_names);

    for (auto& [pname, pinfo] : ppr_profiles_by_name) {
      auto pkey = storage_.profile_name_to_profile_key_.at(pname);
      storage_.used_profiles_.insert(pkey);

      // convert walk_duration from minutes to seconds
      storage_.profile_key_to_profile_info_.insert(
          std::pair<profile_key_t, ppr::profile_info>(
              pkey, ppr_profiles_by_name.at(pname)));
      storage_.profile_key_to_profile_info_.at(pkey).profile_.duration_limit_ =
          ::motis::MAX_WALK_TIME;

      // build profile_name to idx map in nigiri::tt
      storage_.tt_.profiles_.insert({pname, storage_.tt_.profiles_.size()});
    }
    assert(storage_.tt_.profiles_.size() == storage_.used_profiles_.size());
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
    auto osm_extracted_platforms = extract_platforms_from_osm_file(osm_path_);

    LOG(ml::info) << "Writing OSM Platforms to DB.";
    storage_.add_new_platforms(osm_extracted_platforms);
  }

  // -- location to osm matching --
  void match_and_save_matches() {
    progress_tracker_->status("Match Locations and OSM Platforms");
    ml::scoped_timer const timer{
        "Matching timetable locations and osm platforms."};

    auto matcher =
        distance_matcher(storage_.get_matching_data(),
                         {max_matching_dist_, max_bus_stop_matching_dist_});
    auto mrs = matcher.matching();

    LOG(ml::info) << "Writing Matchings to DB.";
    storage_.add_new_matching_results(mrs);
  }

  // -- build transfer requests --
  void build_and_save_transfer_requests(bool const old_to_old = false) {
    progress_tracker_->status("Generating Transfer Requests.");
    ml::scoped_timer const timer{"Generating Transfer Requests."};

    auto treqs_k = generate_transfer_requests_keys(
        storage_.get_transfer_request_keys_generation_data(), {old_to_old});

    LOG(ml::info) << "Writing Transfer Requests (Keys) to DB.";
    storage_.add_new_transfer_requests_keys(treqs_k);
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

    auto matches = storage_.get_all_matchings();

    auto treqs = to_transfer_requests(treqs_k, matches);
    auto trs = route_multiple_requests(treqs, rg,
                                       storage_.profile_key_to_profile_info_);
    storage_.add_new_transfer_results(trs);
  }

  storage storage_;

  // initialize matching limits
  double max_matching_dist_{400};
  double max_bus_stop_matching_dist_{120};

  // initialize progress tracker (ptr)
  utl::progress_tracker_ptr progress_tracker_{
      utl::get_active_progress_tracker()};
};  // namespace motis::transfers

transfers::transfers() : module("Transfers", "transfers") {}

transfers::~transfers() = default;

fs::path transfers::module_data_dir() const {
  return get_data_directory() / "transfers";
}

fs::path transfers::db_file() const {
  return module_data_dir() / "transfers.db";
}

void transfers::import(motis::module::import_dispatcher& reg) {
  std::make_shared<mm::event_collector>(
      get_data_directory().generic_string(), "transfers", reg,
      [this](mm::event_collector::dependencies_map_t const& dependencies,
             mm::event_collector::publish_fn_t const&) {
        using import::FileEvent;
        using import::NigiriEvent;
        using import::PPREvent;

        auto const dir = get_data_directory() / "transfers";
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
        impl_->osm_path_ = fs::path{osm_path};
        impl_->ppr_rg_path_ = ppr->graph_path()->str();
        impl_->nigiri_dump_path_ =
            get_data_directory() / "nigiri" / fmt::to_string(nigiri->hash());

        {
          ml::scoped_timer const timer{"Transfer Import"};
          if (!fs::exists(dir / "import.ini")) {
            LOG(ml::info) << "Transfers: Full Import.";
            impl_->full_import();
            import_successful_ = true;
          } else {
            impl_->old_import_state_ =
                mm::read_ini<import_state>(dir / "import.ini");
            if (impl_->old_import_state_.value() != impl_->new_import_state_) {
              LOG(ml::info) << "Transfers: Maybe Partial Import.";
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

}  // namespace motis::transfers
