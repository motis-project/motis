#include "motis/transfers/transfers.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/core/common/constants.h"
#include "motis/core/common/logging.h"
#include "motis/module/event_collector.h"
#include "motis/module/ini_io.h"

#include "transfers/storage/updater.h"
#include "transfers/types.h"

#include "ppr/profiles/parse_search_profile.h"
#include "ppr/routing/search_profile.h"

#include "nigiri/timetable.h"

namespace fs = std::filesystem;
namespace ml = motis::logging;
namespace mm = motis::module;
namespace n = ::nigiri;
namespace t = ::transfers;

namespace pp = ::ppr::profiles;
namespace pr = ::ppr::routing;

namespace motis::transfers {

inline std::string read_file(std::string const& path) {
  std::ifstream f(path);
  std::stringstream ss;
  f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  ss.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  ss << f.rdbuf();
  return ss.str();
}

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
                std::map<std::string, pr::search_profile> const& ppr_profiles,
                t::storage_updater_config const& config)
      : storage_updater_(tt, config) {
    load_ppr_profiles(ppr_profiles);
  };

  t::first_update get_first_update() {
    utl::verify(old_import_state_.has_value(), "no old import state given.");
    auto const old_import_state = old_import_state_.value();

    auto fup = t::first_update::kNoUpdate;
    if (old_import_state.ppr_profiles_hash_ !=
        new_import_state_.ppr_profiles_hash_) {
      fup = t::first_update::kProfiles;
    }

    if (old_import_state.nigiri_hash_ != new_import_state_.nigiri_hash_) {
      fup = t::first_update::kTimetable;
    }

    if (old_import_state.osm_path_ != new_import_state_.osm_path_) {
      fup = t::first_update::kOSM;
    }

    return fup;
  }

  t::routing_type get_routing_type(t::first_update const fup) {
    utl::verify(old_import_state_.has_value(), "no old import state given.");
    auto const old_import_state = old_import_state_.value();

    auto rt = t::routing_type::kNoRouting;
    // define routing type
    if (old_import_state.ppr_graph_hash_ != new_import_state_.ppr_graph_hash_) {
      rt = t::routing_type::kFullRouting;
    }

    if (fup != t::first_update::kNoUpdate &&
        rt == t::routing_type::kNoRouting) {
      rt = t::routing_type::kPartialRouting;
    }

    return rt;
  }

  t::storage_updater storage_updater_;

  std::optional<import_state> old_import_state_;
  import_state new_import_state_;

private:
  // -- helper --
  void load_ppr_profiles(std::map<std::string, pr::search_profile> const&
                             search_profiles_by_name) {
    auto profile_names = std::vector<t::string>{};

    for (auto const& [pname, profile] : search_profiles_by_name) {
      profile_names.emplace_back(pname);
    }

    storage_updater_.storage_.add_new_profiles(profile_names);

    for (auto& [pname, profile] : search_profiles_by_name) {
      auto pkey =
          storage_updater_.storage_.profile_name_to_profile_key_.at(pname);
      storage_updater_.storage_.used_profiles_.insert(pkey);

      // convert walk_duration from minutes to seconds
      storage_updater_.storage_.profile_key_to_search_profile_.emplace(pkey,
                                                                       profile);
      storage_updater_.storage_.profile_key_to_search_profile_.at(pkey)
          .duration_limit_ = ::motis::MAX_WALK_TIME;

      // build profile_name to idx map in nigiri::tt
      storage_updater_.storage_.tt_.profiles_.insert(
          {pname, storage_updater_.storage_.tt_.profiles_.size()});
    }
    assert(storage_updater_.storage_.tt_.profiles_.size() ==
           storage_updater_.storage_.used_profiles_.size());
  }

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

        auto ppr_profiles = std::map<std::string, pr::search_profile>{};
        for (auto const& p : *ppr->profiles()) {
          auto const content = read_file(p->path()->str());
          auto const profile = pp::parse_search_profile(content);
          ppr_profiles.emplace(p->name()->str(), profile);
        }

        fs::create_directories(dir);
        auto updater_config = t::storage_updater_config{
            .db_file_path_ = db_file(),
            .db_max_size_ = db_max_size_,
            .osm_path_ = osm_path,
            .ppr_rg_path_ = ppr->graph_path()->str(),
            .nigiri_dump_path_ = get_data_directory() / "nigiri" /
                                 fmt::to_string(nigiri->hash()),
            .max_matching_dist_ = 400,
            .max_bus_stop_matching_dist_ = 120,
            .rg_config_ = {.edge_rtree_size_ = edge_rtree_max_size_,
                           .area_rtree_size_ = area_rtree_max_size_,
                           .lock_rtree_ = lock_rtree_}};
        impl_ = std::make_unique<impl>(
            *get_shared_data<n::timetable*>(
                to_res_id(mm::global_res_id::NIGIRI_TIMETABLE)),
            ppr_profiles, updater_config);

        impl_->new_import_state_ = new_import_state;

        {
          ml::scoped_timer const timer{"Transfer Import"};
          if (!fs::exists(dir / "import.ini")) {
            LOG(ml::info) << "Transfers: Full Import.";
            impl_->storage_updater_.full_update();
            import_successful_ = true;
          } else {
            impl_->old_import_state_ =
                mm::read_ini<import_state>(dir / "import.ini");
            if (impl_->old_import_state_.value() != impl_->new_import_state_) {
              LOG(ml::info) << "Transfers: Maybe Partial Import.";
              auto first = impl_->get_first_update();
              auto routing = impl_->get_routing_type(first);
              impl_->storage_updater_.partial_update(first, routing);
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
