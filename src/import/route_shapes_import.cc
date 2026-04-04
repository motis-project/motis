#include "motis/import/route_shapes_import.h"

#include <string>

#include "cista/hashing.h"

#include "nigiri/shapes_storage.h"

#include "motis/data.h"
#include "motis/route_shapes.h"

using namespace std::string_literals;

namespace motis {

namespace fs = std::filesystem;

bool is_route_shapes_task_enabled(config const& c) {
  return c.timetable_
      .transform([](auto&& x) { return x.route_shapes_.has_value(); })
      .value_or(false);
}

auto get_route_shapes_settings(config const& c) {
  return c.timetable_.value_or(config::timetable{})
      .route_shapes_.value_or(config::timetable::route_shapes{});
}

meta_entry_t get_route_shapes_clasz_hash(
    std::array<bool, nigiri::kNumClasses> const& route_shapes_clasz_enabled) {
  auto route_shapes_clasz_hash =
      std::pair{"route_shapes_clasz"s, cista::BASE_HASH};
  for (auto const& b : route_shapes_clasz_enabled) {
    route_shapes_clasz_hash.second =
        cista::build_seeded_hash(route_shapes_clasz_hash.second, b);
  }
  return route_shapes_clasz_hash;
}

std::array<bool, nigiri::kNumClasses> route_shapes_import::get_clasz_enabled(
    config const& c) {
  auto a = std::array<bool, nigiri::kNumClasses>{};
  a.fill(true);
  for (auto const& [clasz, enabled] :
       get_route_shapes_settings(c).clasz_.value_or(
           std::map<std::string, bool>{})) {
    a[static_cast<unsigned>(nigiri::to_clasz(clasz))] = enabled;
  }
  return a;
}

bool route_shapes_import::get_reuse_shapes_cache(fs::path const& data_path,
                                                 config const& c,
                                                 dataset_hashes const& h) {
  auto const shape_cache_path = data_path / "routed_shapes_cache.mdb";
  auto const existing_rs_hashes = read_hashes(data_path, "route_shapes");
  auto const route_shapes_reuse_old_osm_data =
      get_route_shapes_settings(c).cache_reuse_old_osm_data_;

  return fs::exists(shape_cache_path) &&
         (existing_rs_hashes.find("routed_shapes_ver") !=
              end(existing_rs_hashes) &&
          existing_rs_hashes.at("routed_shapes_ver") ==
              routed_shapes_version().second) &&
         (route_shapes_reuse_old_osm_data ||
          (existing_rs_hashes.find(h.osm_.first) != end(existing_rs_hashes) &&
           existing_rs_hashes.at(h.osm_.first) == h.osm_.second &&
           existing_rs_hashes.find("cache_reuse_old_osm_data") !=
               end(existing_rs_hashes) &&
           (existing_rs_hashes.at("cache_reuse_old_osm_data") ==
                static_cast<std::uint64_t>(route_shapes_reuse_old_osm_data) ||
            existing_rs_hashes.at("cache_reuse_old_osm_data") == 0)));
}

bool route_shapes_import::get_keep_routed_shape_data(fs::path const& data_path,
                                                     config const& c,
                                                     dataset_hashes const& h) {
  return !is_route_shapes_task_enabled(c) ||
         get_reuse_shapes_cache(data_path, c, h);
}

void route_shapes_import::cleanup_stale_cache(fs::path const& data_path) {
  auto ec = std::error_code{};
  auto const shape_cache_path = data_path / "routed_shapes_cache.mdb";
  auto const shape_cache_lock_path =
      fs::path{shape_cache_path.generic_string() + "-lock"};
  fs::remove(shape_cache_path, ec);
  fs::remove(shape_cache_lock_path, ec);
}

route_shapes_import::route_shapes_import(
    fs::path const& data_path,
    config const& c,
    dataset_hashes const& h)
    : task{"route_shapes",
           data_path,
           c,
           {h.tt_,
            h.osm_,
            osr_version(),
            n_version(),
            routed_shapes_version(),
            get_route_shapes_clasz_hash(get_clasz_enabled(c)),
            {"route_shapes_mode",
             static_cast<std::uint64_t>(
                 c.timetable_.value_or(config::timetable{})
                     .route_shapes_.value_or(config::timetable::route_shapes{})
                     .mode_)},
            {"cache_reuse_old_osm_data",
             c.timetable_.value_or(config::timetable{})
                 .route_shapes_.value_or(config::timetable::route_shapes{})
                 .cache_reuse_old_osm_data_}}},
      route_shapes_clasz_enabled_{get_clasz_enabled(c)},
      reuse_shapes_cache_{get_reuse_shapes_cache(data_path, c, h)} {
  if (!reuse_shapes_cache_) {
    route_shapes_import::cleanup_stale_cache(data_path);
  }
}

route_shapes_import::~route_shapes_import() = default;

void route_shapes_import::run() {
  auto d = data{data_path_, false};
  d.load_osr();
  d.load_tt("tt.bin");

  auto const shape_cache_path = data_path_ / "routed_shapes_cache.mdb";

  if (reuse_shapes_cache_) {
    std::clog << "loading existing shape cache from " << shape_cache_path
              << "\n";
  } else {
    std::clog << "creating new shape cache\n";
  }

  auto shape_cache = std::make_unique<motis::shape_cache>(
      shape_cache_path, c_.timetable_->route_shapes_->cache_db_size_);
  auto shapes = std::make_unique<nigiri::shapes_storage>(
      data_path_, cista::mmap::protection::MODIFY, reuse_shapes_cache_);

  route_shapes(*d.w_, *d.l_, *d.tt_, *shapes, *c_.timetable_->route_shapes_,
               route_shapes_clasz_enabled_, shape_cache.get());
}

bool route_shapes_import::is_enabled() const {
  return is_route_shapes_task_enabled(c_);
}

}  // namespace motis
