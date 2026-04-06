#include "motis/import.h"

#include <fstream>
#include <map>
#include <ostream>
#include <tuple>
#include <vector>

#include "fmt/ranges.h"

#include "cista/free_self_allocated.h"
#include "cista/io.h"

#include "adr/area_database.h"

#include "utl/erase_if.h"
#include "utl/read_file.h"
#include "utl/to_vec.h"

#include "tiles/db/clear_database.h"
#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/feature_pack.h"
#include "tiles/db/pack_file.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/osm/load_coastlines.h"
#include "tiles/osm/load_osm.h"

#include "nigiri/loader/assistance.h"
#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/clasz.h"
#include "nigiri/common/parse_date.h"
#include "nigiri/routing/tb/preprocess.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"

#include "osr/extract/extract.h"
#include "osr/lookup.h"
#include "osr/ways.h"

#include "adr/adr.h"
#include "adr/formatter.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "motis/adr_extend_tt.h"
#include "motis/clog_redirect.h"
#include "motis/compute_footpaths.h"
#include "motis/data.h"
#include "motis/hashes.h"
#include "motis/route_shapes.h"
#include "motis/tag_lookup.h"
#include "motis/tt_location_rtree.h"

namespace fs = std::filesystem;
namespace n = nigiri;
namespace nl = nigiri::loader;
using namespace std::string_literals;
using std::chrono_literals::operator""min;
using std::chrono_literals::operator""h;

namespace motis {

struct task {
  friend std::ostream& operator<<(std::ostream& out, task const& t) {
    out << t.name_ << " ";
    if (t.done_) {
      out << "done";
    } else if (!t.should_run_) {
      out << "disabled";
    } else if (utl::all_of(t.dependencies_,
                           [](task const* t) { return t->done_; })) {
      out << "ready";
    } else {
      out << "waiting for ";
      auto first = true;
      for (auto const& dep : t.dependencies_) {
        if (!first) {
          out << ", ";
        }
        first = false;
        out << dep->name_;
      }
    }
    return out;
  }

  bool ready_for_load(fs::path const& data_path) {
    auto const existing = read_hashes(data_path, name_);
    if (existing != hashes_) {
      std::cout << name_ << "\n"
                << "  existing: " << to_str(existing) << "\n"
                << "  current: " << to_str(hashes_) << "\n";
    }
    return existing == hashes_;
  }

  void run(fs::path const& data_path) {
    auto const pt = utl::activate_progress_tracker(name_);
    auto const redirect = clog_redirect{
        (data_path / "logs" / (name_ + ".txt")).generic_string().c_str()};
    run_();
    write_hashes(data_path, name_, hashes_);
    pt->out_ = 100;
    pt->status("FINISHED");
    done_ = true;
  }

  std::string name_;
  std::vector<task*> dependencies_;
  bool should_run_;
  std::function<void()> run_;
  meta_t hashes_;
  bool done_{false};
  utl::progress_tracker_ptr pt_{};
};

}  // namespace motis

template <>
struct fmt::formatter<motis::task> : fmt::ostream_formatter {};

namespace motis {

cista::hash_t hash_file(fs::path const& p) {
  auto const str = p.generic_string();
  if (str.starts_with("\nfunction") || str.starts_with("\n#")) {
    return cista::hash(str);
  } else if (fs::is_directory(p)) {
    auto h = cista::BASE_HASH;
    auto entries = std::vector<std::tuple<std::string, std::uint64_t,
                                          std::filesystem::file_time_type>>{};
    for (auto const& entry : fs::recursive_directory_iterator{p}) {
      auto ec = std::error_code{};
      entries.emplace_back(fs::relative(entry.path(), p, ec).generic_string(),
                           entry.is_regular_file(ec) ? entry.file_size(ec) : 0U,
                           fs::last_write_time(entry.path(), ec));
    }
    utl::sort(entries);
    for (auto const& [rel, size, modified_ts] : entries) {
      h = cista::hash_combine(h, cista::hash(rel), size,
                              modified_ts.time_since_epoch().count());
    }
    return h;
  } else {
    auto const mmap = cista::mmap{str.c_str(), cista::mmap::protection::READ};
    return cista::hash_combine(
        cista::hash(mmap.view().substr(
            0U, std::min(mmap.size(),
                         static_cast<std::size_t>(50U * 1024U * 1024U)))),
        mmap.size());
  }
}

void import(config const& c,
            fs::path const& data_path,
            std::optional<std::vector<std::string>> const& task_filter) {
  c.verify_input_files_exist();

  auto ec = std::error_code{};
  fs::create_directories(data_path / "logs", ec);
  fs::create_directories(data_path / "meta", ec);
  {
    auto cfg = std::ofstream{(data_path / "config.yml").generic_string()};
    cfg.exceptions(std::ios_base::badbit | std::ios_base::eofbit);
    cfg << c << "\n";
    cfg.close();
  }

  clog_redirect::set_enabled(true);

  auto tt_hash = std::pair{"timetable"s, cista::BASE_HASH};
  if (c.timetable_.has_value()) {
    auto& h = tt_hash.second;
    auto const& t = *c.timetable_;

    for (auto const& [_, d] : t.datasets_) {
      h = cista::build_seeded_hash(
          h, c.osr_footpath_, hash_file(d.path_), d.default_bikes_allowed_,
          d.default_cars_allowed_, d.clasz_bikes_allowed_,
          d.clasz_cars_allowed_, d.default_timezone_, d.extend_calendar_);
      if (d.script_.has_value()) {
        h = cista::build_seeded_hash(h, hash_file(*d.script_));
      }
    }

    h = cista::build_seeded_hash(
        h, t.first_day_, t.num_days_, t.with_shapes_, t.adjust_footpaths_,
        t.merge_dupes_intra_src_, t.merge_dupes_inter_src_,
        t.link_stop_distance_, t.update_interval_, t.incremental_rt_update_,
        t.max_footpath_length_, t.default_timezone_, t.assistance_times_);
  }

  auto osm_hash = std::pair{"osm"s, cista::BASE_HASH};
  if (c.osm_.has_value()) {
    osm_hash.second = hash_file(*c.osm_);
  }

  auto const elevation_dir =
      c.get_street_routing()
          .and_then([](config::street_routing const& sr) {
            return sr.elevation_data_dir_;
          })
          .value_or(fs::path{});
  auto elevation_dir_hash = std::pair{"elevation_dir"s, cista::BASE_HASH};
  if (!elevation_dir.empty() && fs::exists(elevation_dir)) {
    auto files = std::vector<std::string>{};
    for (auto const& f : fs::recursive_directory_iterator(elevation_dir)) {
      if (f.is_regular_file()) {
        files.emplace_back(f.path().relative_path().string());
      }
    }
    std::ranges::sort(files);
    auto& h = elevation_dir_hash.second;
    for (auto const& f : files) {
      h = cista::build_seeded_hash(h, f);
    }
  }

  auto tiles_hash = std::pair{"tiles_profile", cista::BASE_HASH};
  if (c.tiles_.has_value()) {
    auto& h = tiles_hash.second;
    h = cista::build_hash(hash_file(c.tiles_->profile_), c.tiles_->db_size_);
    if (c.tiles_->coastline_.has_value()) {
      h = cista::hash_combine(h, hash_file(*c.tiles_->coastline_));
    }
  }

  auto const to_clasz_bool_array =
      [&](bool const default_allowed,
          std::optional<std::map<std::string, bool>> const& clasz_allowed) {
        auto a = std::array<bool, n::kNumClasses>{};
        a.fill(default_allowed);
        if (clasz_allowed.has_value()) {
          for (auto const& [clasz, allowed] : *clasz_allowed) {
            a[static_cast<unsigned>(n::to_clasz(clasz))] = allowed;
          }
        }
        return a;
      };

  auto const route_shapes_clasz_enabled = to_clasz_bool_array(
      true, c.timetable_.value_or(config::timetable{})
                .route_shapes_.value_or(config::timetable::route_shapes{})
                .clasz_);
  auto route_shapes_clasz_hash =
      std::pair{"route_shapes_clasz"s, cista::BASE_HASH};
  for (auto const& b : route_shapes_clasz_enabled) {
    route_shapes_clasz_hash.second =
        cista::build_seeded_hash(route_shapes_clasz_hash.second, b);
  }

  auto const shape_cache_path = data_path / "routed_shapes_cache.mdb";
  auto const shape_cache_lock_path =
      fs::path{shape_cache_path.generic_string() + "-lock"};
  auto const route_shapes_task_enabled =
      c.timetable_
          .transform([](auto&& x) { return x.route_shapes_.has_value(); })
          .value_or(false);
  auto const existing_rs_hashes = read_hashes(data_path, "route_shapes");
  auto const route_shapes_reuse_old_osm_data =
      c.timetable_.value_or(config::timetable{})
          .route_shapes_.value_or(config::timetable::route_shapes{})
          .cache_reuse_old_osm_data_;
  auto const reuse_shapes_cache =
      // cache must exist (handles case where files were deleted manually)
      fs::exists(shape_cache_path) &&
      // and have the same routed_shapes_ver
      (existing_rs_hashes.find("routed_shapes_ver") !=
           end(existing_rs_hashes) &&
       existing_rs_hashes.at("routed_shapes_ver") ==
           routed_shapes_version().second) &&
      // if route_shapes_reuse_old_osm_data, we can reuse any data
      // otherwise only if the osm data is the same
      (route_shapes_reuse_old_osm_data ||
       (existing_rs_hashes.find(osm_hash.first) != end(existing_rs_hashes) &&
        existing_rs_hashes.at(osm_hash.first) == osm_hash.second &&
        // cache_reuse_old_osm_data flag must be the same or changed from 0->1
        // otherwise cache may contain old data from previous runs
        existing_rs_hashes.find("cache_reuse_old_osm_data") !=
            end(existing_rs_hashes) &&
        (existing_rs_hashes.at("cache_reuse_old_osm_data") ==
             static_cast<std::uint64_t>(route_shapes_reuse_old_osm_data) ||
         existing_rs_hashes.at("cache_reuse_old_osm_data") == 0)));

  auto const keep_routed_shape_data =
      !route_shapes_task_enabled || reuse_shapes_cache;

  if (!keep_routed_shape_data) {
    fs::remove(shape_cache_path, ec);
    fs::remove(shape_cache_lock_path, ec);
  }

  auto osr = task{"osr",
                  {},
                  c.use_street_routing(),
                  [&]() {
                    osr::extract(true, fs::path{*c.osm_}, data_path / "osr",
                                 elevation_dir);
                  },
                  {osm_hash, osr_version(), elevation_dir_hash}};

  auto adr = task{"adr",
                  {},
                  c.geocoding_ || c.reverse_geocoding_,
                  [&]() {
                    if (!c.osm_) {
                      return;
                    }
                    adr::extract(*c.osm_, data_path / "adr", data_path / "adr");
                  },
                  {osm_hash,
                   adr_version(),
                   {"geocoding", c.geocoding_},
                   {"reverse_geocoding", c.reverse_geocoding_}}};

  auto tt = task{
      "tt",
      {},
      c.timetable_.has_value(),
      [&]() {
        auto const& t = *c.timetable_;

        auto const first_day =
            n::parse_date(t.first_day_) - std::chrono::days{1};
        auto const interval = n::interval<date::sys_days>{
            first_day, first_day + std::chrono::days{t.num_days_ + 1U}};

        auto assistance = std::unique_ptr<nl::assistance_times>{};
        if (t.assistance_times_.has_value()) {
          auto const f =
              cista::mmap{t.assistance_times_->generic_string().c_str(),
                          cista::mmap::protection::READ};
          assistance = std::make_unique<nl::assistance_times>(
              nl::read_assistance(f.view()));
        }

        auto shapes = std::unique_ptr<n::shapes_storage>{};
        if (t.with_shapes_) {
          shapes = std::make_unique<n::shapes_storage>(
              data_path, cista::mmap::protection::WRITE,
              keep_routed_shape_data);
        }

        auto tags = cista::wrapped{cista::raw::make_unique<tag_lookup>()};
        auto tt = cista::wrapped{cista::raw::make_unique<n::timetable>(nl::load(
            utl::to_vec(
                t.datasets_,
                [&, src = n::source_idx_t{}](
                    std::pair<std::string, config::timetable::dataset> const&
                        x) mutable -> nl::timetable_source {
                  auto const& [tag, dc] = x;
                  tags->add(src++, tag);
                  return {
                      tag,
                      dc.path_,
                      {.link_stop_distance_ = t.link_stop_distance_,
                       .default_tz_ = dc.default_timezone_.value_or(
                           t.default_timezone_.value_or("")),
                       .bikes_allowed_default_ = to_clasz_bool_array(
                           dc.default_bikes_allowed_, dc.clasz_bikes_allowed_),
                       .cars_allowed_default_ = to_clasz_bool_array(
                           dc.default_cars_allowed_, dc.clasz_cars_allowed_),
                       .extend_calendar_ = dc.extend_calendar_,
                       .user_script_ =
                           dc.script_
                               .and_then([](std::string const& path) {
                                 if (path.starts_with("\nfunction")) {
                                   return std::optional{path};
                                 }
                                 return std::optional{std::string{
                                     cista::mmap{path.c_str(),
                                                 cista::mmap::protection::READ}
                                         .view()}};
                               })
                               .value_or("")}};
                }),
            {.adjust_footpaths_ = t.adjust_footpaths_,
             .merge_dupes_intra_src_ = t.merge_dupes_intra_src_,
             .merge_dupes_inter_src_ = t.merge_dupes_inter_src_,
             .max_footpath_length_ = t.max_footpath_length_},
            interval, assistance.get(), shapes.get(), false))};

        tt->write(data_path / "tt.bin");
        tags->write(data_path / "tags.bin");
        std::ofstream(data_path / "timetable_metrics.json")
            << to_str(n::get_metrics(*tt), *tt);
      },
      {tt_hash, n_version()}};

  auto tbd = task{"tbd",
                  {&tt},
                  c.timetable_.has_value() && c.timetable_->tb_,
                  [&]() {
                    auto d = data{data_path};
                    d.load_tt("tt.bin");
                    cista::write(
                        data_path / "tbd.bin",
                        n::routing::tb::preprocess(*d.tt_, n::kDefaultProfile));
                  },
                  {tt_hash, n_version(), tbd_version()}};

  auto adr_extend = task{
      "adr_extend",
      c.osm_.has_value() ? std::vector<task*>{&adr} : std::vector<task*>{},
      c.timetable_.has_value() && (c.geocoding_ || c.reverse_geocoding_),
      [&]() {
        auto d = data{data_path};
        d.load_tt("tt.bin");
        if (c.osm_) {
          d.t_ = adr::read(data_path / "adr" / "t.bin");
          d.tc_ = std::make_unique<adr::cache>(d.t_->strings_.size(), 100U);
          d.f_ = std::make_unique<adr::formatter>();
        }

        auto const area_db = d.t_ ? (std::optional<adr::area_database>{
                                        std::in_place, data_path / "adr",
                                        cista::mmap::protection::READ})
                                  : std::nullopt;
        if (!d.t_) {
          d.t_ = cista::wrapped<adr::typeahead>{
              cista::raw::make_unique<adr::typeahead>()};
        }
        auto const location_extra_place = adr_extend_tt(
            *d.tt_, area_db.has_value() ? &*area_db : nullptr, *d.t_);
        auto ec = std::error_code{};
        std::filesystem::create_directories(data_path / "adr", ec);
        cista::write(data_path / "adr" / "t_ext.bin", *d.t_);
        cista::write(data_path / "adr" / "location_extra_place.bin",
                     location_extra_place);
        {
          auto r =
              adr::reverse{data_path / "adr", cista::mmap::protection::WRITE};
          r.build_rtree(*d.t_);
          r.write();
        }

        cista::free_self_allocated(d.t_.get());
      },
      {tt_hash,
       osm_hash,
       adr_version(),
       adr_ext_version(),
       n_version(),
       {"geocoding", c.geocoding_},
       {"reverse_geocoding", c.reverse_geocoding_}}};

  auto osr_footpath_settings_hash =
      meta_entry_t{"osr_footpath_settings", cista::BASE_HASH};
  if (c.timetable_) {
    auto& h = osr_footpath_settings_hash.second;
    h = cista::hash_combine(h, c.timetable_->use_osm_stop_coordinates_,
                            c.timetable_->extend_missing_footpaths_,
                            c.timetable_->max_matching_distance_,
                            c.timetable_->max_footpath_length_);
  }
  auto osr_footpath = task{
      "osr_footpath",
      {&tt, &osr},
      c.osr_footpath_ && c.timetable_,
      [&]() {
        auto d = data{data_path};
        d.load_tt("tt.bin");
        d.load_osr();

        auto const profiles = std::vector<routed_transfers_settings>{
            {.profile_ = osr::search_profile::kFoot,
             .profile_idx_ = n::kFootProfile,
             .max_matching_distance_ = c.timetable_->max_matching_distance_,
             .extend_missing_ = c.timetable_->extend_missing_footpaths_,
             .max_duration_ = c.timetable_->max_footpath_length_ * 1min},
            {.profile_ = osr::search_profile::kWheelchair,
             .profile_idx_ = n::kWheelchairProfile,
             .max_matching_distance_ = 8.0,
             .max_duration_ = c.timetable_->max_footpath_length_ * 1min},
            {.profile_ = osr::search_profile::kCar,
             .profile_idx_ = n::kCarProfile,
             .max_matching_distance_ = 250.0,
             .max_duration_ = 8h,
             .is_candidate_ = [&](n::location_idx_t const l) {
               return utl::any_of(d.tt_->location_routes_[l], [&](auto r) {
                 return d.tt_->has_car_transport(r);
               });
             }}};
        auto const elevator_footpath_map = compute_footpaths(
            *d.w_, *d.l_, *d.pl_, *d.tt_, d.elevations_.get(),
            c.timetable_->use_osm_stop_coordinates_, profiles);

        cista::write(data_path / "elevator_footpath_map.bin",
                     elevator_footpath_map);
        d.tt_->write(data_path / "tt_ext.bin");

        cista::free_self_allocated(d.tt_.get());
      },
      {tt_hash, osm_hash, osr_footpath_settings_hash, osr_version(),
       osr_footpath_version(), n_version()}};

  auto matches = task{
      "matches",
      {&tt, &osr, &osr_footpath},
      c.timetable_ && c.use_street_routing(),
      [&]() {
        auto d = data{data_path};
        d.load_tt(c.osr_footpath_ ? "tt_ext.bin" : "tt.bin");
        d.load_osr();

        auto const progress_tracker = utl::get_active_progress_tracker();
        progress_tracker->status("Prepare Platform Matches").out_bounds(0, 30);
        cista::write(data_path / "matches.bin",
                     get_matches(*d.tt_, *d.pl_, *d.w_));

        d.load_matches();
        if (c.timetable_.value().preprocess_max_matching_distance_ > 0.0) {
          progress_tracker->status("Prepare Platform Way Matches")
              .out_bounds(30, 100);
          way_matches_storage{
              data_path, cista::mmap::protection::WRITE,
              c.timetable_.value().preprocess_max_matching_distance_}
              .preprocess_osr_matches(*d.tt_, *d.pl_, *d.w_, *d.l_,
                                      *d.matches_);
        }
      },
      {tt_hash, osm_hash, osr_version(), n_version(), matches_version(),
       std::pair{"way_matches",
                 cista::build_hash(c.timetable_.value_or(config::timetable{})
                                       .preprocess_max_matching_distance_)}}};

  auto route_shapes_task = task{
      "route_shapes",
      {&tt, &osr},
      route_shapes_task_enabled,
      [&]() {
        auto d = data{data_path};
        d.load_tt("tt.bin");
        d.load_osr();

        auto shape_cache = std::unique_ptr<motis::shape_cache>{};
        if (reuse_shapes_cache) {
          std::clog << "loading existing shape cache from " << shape_cache_path
                    << "\n";
        } else {
          std::clog << "creating new shape cache\n";
        }
        shape_cache = std::make_unique<motis::shape_cache>(
            shape_cache_path, c.timetable_->route_shapes_->cache_db_size_);

        // re-open in write mode
        // this needs to be done in two steps, because the files need to be
        // closed first, before they can be re-opened in write mode (at
        // least on Windows)
        d.shapes_ = {};
        auto shapes = n::shapes_storage{
            data_path, cista::mmap::protection::MODIFY, reuse_shapes_cache};
        route_shapes(*d.w_, *d.l_, *d.tt_, shapes, *c.timetable_->route_shapes_,
                     route_shapes_clasz_enabled, shape_cache.get());
      },
      {tt_hash,
       osm_hash,
       osr_version(),
       n_version(),
       routed_shapes_version(),
       route_shapes_clasz_hash,
       {"route_shapes_mode",
        static_cast<std::uint64_t>(
            c.timetable_.value_or(config::timetable{})
                .route_shapes_.value_or(config::timetable::route_shapes{})
                .mode_)},
       {"cache_reuse_old_osm_data",
        c.timetable_.value_or(config::timetable{})
            .route_shapes_.value_or(config::timetable::route_shapes{})
            .cache_reuse_old_osm_data_}}};

  auto tiles = task{
      "tiles",
      {},
      c.tiles_.has_value(),
      [&]() {
        auto const progress_tracker = utl::get_active_progress_tracker();

        auto const dir = data_path / "tiles";
        auto const path = (dir / "tiles.mdb").string();

        auto ec = std::error_code{};
        fs::create_directories(data_path / "tiles", ec);

        progress_tracker->status("Clear Database");
        ::tiles::clear_database(path, c.tiles_->db_size_);
        ::tiles::clear_pack_file(path.c_str());

        auto db_env =
            ::tiles::make_tile_database(path.c_str(), c.tiles_->db_size_);
        ::tiles::tile_db_handle db_handle{db_env};
        ::tiles::pack_handle pack_handle{path.c_str()};

        {
          ::tiles::feature_inserter_mt inserter{
              ::tiles::dbi_handle{db_handle, db_handle.features_dbi_opener()},
              pack_handle};

          if (c.tiles_->coastline_.has_value()) {
            progress_tracker->status("Load Coastlines").out_bounds(0, 20);
            ::tiles::load_coastlines(db_handle, inserter,
                                     c.tiles_->coastline_->generic_string());
          }

          progress_tracker->status("Load Features").out_bounds(20, 70);
          ::tiles::load_osm(db_handle, inserter, c.osm_->generic_string(),
                            c.tiles_->profile_.generic_string(),
                            dir.generic_string(), c.tiles_->flush_threshold_);
        }

        progress_tracker->status("Pack Features").out_bounds(70, 90);
        ::tiles::pack_features(db_handle, pack_handle);

        progress_tracker->status("Prepare Tiles").out_bounds(90, 100);
        ::tiles::prepare_tiles(db_handle, pack_handle, 10);
      },
      {tiles_version(), osm_hash, tiles_hash}};

  auto all_tasks = std::vector{&tiles,        &osr,     &adr,
                               &tt,           &tbd,     &adr_extend,
                               &osr_footpath, &matches, &route_shapes_task};
  auto todo = std::set<task*>{};
  if (task_filter.has_value()) {
    auto q = std::vector<task*>{};
    for (auto const& x : *task_filter) {
      auto const it =
          utl::find_if(all_tasks, [&](task* t) { return t->name_ == x; });
      utl::verify(it != end(all_tasks) && (*it)->should_run_,
                  "task {} not found or disabled", x);
      q.push_back(*it);
    }

    while (!q.empty()) {
      auto const next = q.back();
      q.resize(q.size() - 1);
      todo.insert(next);
      for (auto const& x : next->dependencies_) {
        if (!todo.contains(x)) {
          q.push_back(x);
        }
      }
    }
    fmt::println("running tasks: {}",
                 todo | std::views::transform([](task* t) { return *t; }));
  } else {
    todo.insert(begin(all_tasks), end(all_tasks));
  }

  auto tasks = std::vector<task*>{begin(todo), end(todo)};
  utl::erase_if(tasks, [&](task* t) {
    if (!t->should_run_) {
      t->done_ = true;
      return true;
    }

    if (t->ready_for_load(data_path)) {
      t->done_ = true;
      return true;
    }

    return false;
  });

  for (auto* t : tasks) {
    t->pt_ = utl::activate_progress_tracker(t->name_);
  }

  while (!tasks.empty()) {
    auto const task_it = utl::find_if(tasks, [](task const* t) {
      return utl::all_of(t->dependencies_,
                         [](task const* t) { return t->done_; });
    });
    utl::verify(
        task_it != end(tasks), "no task to run, remaining tasks: {}",
        tasks | std::views::transform([](task const* t) { return *t; }));
    (*task_it)->run(data_path);
    tasks.erase(task_it);
  }
}

}  // namespace motis
