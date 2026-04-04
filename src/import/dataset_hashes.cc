#include "motis/import/dataset_hashes.h"

#include <algorithm>
#include <tuple>
#include <vector>

#include "cista/hashing.h"
#include "cista/mmap.h"

#include "motis/config.h"

namespace motis {

namespace fs = std::filesystem;

cista::hash_t hash_file(fs::path const& p) {
  auto const str = p.generic_string();
  if (str.starts_with("\nfunction") || str.starts_with("\n#")) {
    return cista::hash(str);
  } else if (fs::is_directory(p)) {
    auto h = cista::BASE_HASH;
    auto entries =
        std::vector<std::tuple<std::string, std::uint64_t, fs::file_time_type>>{};
    for (auto const& entry : fs::recursive_directory_iterator{p}) {
      auto ec = std::error_code{};
      entries.emplace_back(fs::relative(entry.path(), p, ec).generic_string(),
                           entry.is_regular_file(ec) ? entry.file_size(ec) : 0U,
                           fs::last_write_time(entry.path(), ec));
    }
    std::ranges::sort(entries);
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

dataset_hashes::dataset_hashes(config const& c)
    : osm_{"osm", cista::BASE_HASH},
      tt_{"timetable", cista::BASE_HASH},
      elevation_{"elevation_dir", cista::BASE_HASH},
      tiles_{"tiles_profile", cista::BASE_HASH} {
  if (c.timetable_.has_value()) {
    auto& h = tt_.second;
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

  if (c.osm_.has_value()) {
    osm_.second = hash_file(*c.osm_);
  }

  auto const elevation_dir =
      c.get_street_routing()
          .and_then([](config::street_routing const& sr) {
            return sr.elevation_data_dir_;
          })
          .value_or(fs::path{});
  if (!elevation_dir.empty() && fs::exists(elevation_dir)) {
    auto files = std::vector<std::string>{};
    for (auto const& f : fs::recursive_directory_iterator(elevation_dir)) {
      if (f.is_regular_file()) {
        files.emplace_back(f.path().relative_path().string());
      }
    }
    std::ranges::sort(files);
    auto& h = elevation_.second;
    for (auto const& f : files) {
      h = cista::build_seeded_hash(h, f);
    }
  }

  if (c.tiles_.has_value()) {
    auto& h = tiles_.second;
    h = cista::build_hash(hash_file(c.tiles_->profile_), c.tiles_->db_size_);
    if (c.tiles_->coastline_.has_value()) {
      h = cista::hash_combine(h, hash_file(*c.tiles_->coastline_));
    }
  }
}

}  // namespace motis
