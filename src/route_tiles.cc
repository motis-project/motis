#include "motis/route_tiles.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "utl/pairwise.h"

#include "geo/box.h"
#include "geo/polyline.h"
#include "geo/polyline_format.h"

#include "osr/lookup.h"

#include "tiles/db/clear_database.h"
#include "tiles/db/feature_inserter_mt.h"
#include "tiles/db/layer_names.h"
#include "tiles/db/pack_file.h"
#include "tiles/db/prepare_tiles.h"
#include "tiles/db/tile_database.h"
#include "tiles/feature/feature.h"
#include "tiles/feature/metadata.h"
#include "tiles/fixed/convert.h"

#include "nigiri/rt/frun.h"

#include "utl/progress_tracker.h"

#include "motis/tag_lookup.h"
#include "motis/types.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {

namespace {

using clock_t = std::chrono::steady_clock;

constexpr auto kOobMatchDistance = 500.0;

struct scoped_timing {
  explicit scoped_timing(std::string name)
      : name_{std::move(name)}, start_{clock_t::now()} {}

  ~scoped_timing() {
    auto const end = clock_t::now();
    auto const ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_)
            .count();
    std::clog << "[route_tiles] " << name_ << ": " << ms << " ms\n";
  }

  std::string name_;
  clock_t::time_point start_;
};

template <typename Container>
std::string join_comma(Container const& values) {
  auto result = std::string{};
  for (auto const& value : values) {
    if (!result.empty()) {
      result += ", ";
    }
    result += value;
  }
  return result;
}

template <typename Container, typename Value>
void add_unique(Container& values, Value&& value) {
  if (std::find(begin(values), end(values), value) == end(values)) {
    values.emplace_back(std::forward<Value>(value));
  }
}

std::int32_t string_to_hash(std::string_view const str) {
  auto hash = std::int32_t{0};
  for (auto const c : str) {
    auto const next =
        static_cast<std::uint32_t>(static_cast<std::uint8_t>(c) +
                                   ((static_cast<std::int64_t>(hash) << 5U) -
                                    static_cast<std::int64_t>(hash)));
    hash = std::bit_cast<std::int32_t>(next);
  }
  return hash;
}

std::uint8_t to_rgb_channel(double const value) {
  auto const scaled =
      std::lround(std::clamp(value, 0.0, 1.0) * static_cast<double>(0xFF));
  return static_cast<std::uint8_t>(scaled);
}

std::string hash_color(std::string_view const name) {
  auto const h = std::abs(string_to_hash(name) % 360);
  auto const hue = static_cast<double>(h) / 60.0;
  auto const saturation = 0.8;
  auto const lightness = 0.6;
  auto const chroma = (1.0 - std::abs((2.0 * lightness) - 1.0)) * saturation;
  auto const x = chroma * (1.0 - std::abs(std::fmod(hue, 2.0) - 1.0));

  auto red = 0.0;
  auto green = 0.0;
  auto blue = 0.0;
  if (hue < 1.0) {
    red = chroma;
    green = x;
  } else if (hue < 2.0) {
    red = x;
    green = chroma;
  } else if (hue < 3.0) {
    green = chroma;
    blue = x;
  } else if (hue < 4.0) {
    green = x;
    blue = chroma;
  } else if (hue < 5.0) {
    red = x;
    blue = chroma;
  } else {
    red = chroma;
    blue = x;
  }

  auto const match = lightness - (chroma / 2.0);
  return fmt::format("#{:02X}{:02X}{:02X}", to_rgb_channel(red + match),
                     to_rgb_channel(green + match),
                     to_rgb_channel(blue + match));
}

bool has_nearby_osm_way(osr::ways const& w,
                        osr::lookup const& l,
                        geo::latlng const& pos) {
  auto const approx_distance_lng_degrees =
      geo::approx_distance_lng_degrees(pos);
  auto const squared_max_dist = std::pow(kOobMatchDistance, 2);
  auto found = false;

  l.find(geo::box{pos, kOobMatchDistance}, [&](osr::way_idx_t const way) {
    auto const [squared_dist, best, segment_idx] =
        geo::approx_squared_distance_to_polyline<
            std::tuple<double, geo::latlng, std::size_t>>(
            pos, w.way_polylines_[way], approx_distance_lng_degrees);
    static_cast<void>(best);
    static_cast<void>(segment_idx);
    found = squared_dist < squared_max_dist;
    return !found;
  });

  return found;
}

bool is_beeline_oob(osr::ways const& w,
                    osr::lookup const& l,
                    geo::latlng const& from,
                    geo::latlng const& to) {
  return !has_nearby_osm_way(w, l, from) || !has_nearby_osm_way(w, l, to);
}

}  // namespace

void import_route_tiles(config const& c, data& d, fs::path const& data_path) {
  auto const total_start = clock_t::now();
  utl::verify(c.route_tiles_.has_value(), "route_tiles config missing");
  utl::verify(d.tt_ && d.tags_ && d.shapes_ && d.w_ && d.l_,
              "route_tiles requires tt/tags/shapes/osr");

  std::clog << "[route_tiles] start\n";

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->out_mod(0.1F);

  auto const dir = data_path / "route_tiles";
  auto const path = (dir / "route_tiles.mdb").string();

  auto ec = std::error_code{};
  fs::create_directories(dir, ec);

  progress_tracker->status("Initialize Route Tile Database").out_bounds(0, 1);
  {
    auto const phase = scoped_timing{"initialize_db"};
    ::tiles::clear_database(path, c.route_tiles_->db_size_);
    ::tiles::clear_pack_file(path.c_str());
  }

  auto db_env =
      ::tiles::make_tile_database(path.c_str(), c.route_tiles_->db_size_);
  ::tiles::tile_db_handle db_handle{db_env};
  ::tiles::pack_handle pack_handle{path.c_str()};

  auto layer_names = ::tiles::layer_names_builder{};
  auto const routes_layer_id = layer_names.get_layer_idx("routes");
  auto const stops_layer_id = layer_names.get_layer_idx("stops");

  auto feature_inserter = ::tiles::feature_inserter_mt{
      ::tiles::dbi_handle{db_handle, db_handle.features_dbi_opener()},
      pack_handle};

  auto const& tt = *d.tt_;
  auto const& tags = *d.tags_;
  auto const* shapes = d.shapes_.get();
  auto const& w = *d.w_;
  auto const& l = *d.l_;
  auto enc = geo::polyline_encoder<6>{};

  struct route_polyline_feature {
    ::tiles::fixed_line line_{};
    std::set<std::string> short_names_{};
    std::string color_{};
    std::optional<std::string> clasz_{};
    bool color_from_timetable_{false};
    bool beeline_{false};
    bool beeline_oob_{false};
  };

  auto polyline_features = hash_map<std::string, route_polyline_feature>{};
  auto stop_locations = std::set<n::location_idx_t>{};

  auto const add_route_segment =
      [&](std::string const& key, ::tiles::fixed_line line,
          std::set<std::string> const& short_names, std::string const& color,
          std::string const& clasz, bool const color_from_timetable,
          bool const beeline, bool const beeline_oob) {
        auto [it, inserted] = polyline_features.try_emplace(key);
        if (inserted) {
          it->second.line_ = std::move(line);
          it->second.color_ = color;
          it->second.clasz_ = clasz;
          it->second.color_from_timetable_ = color_from_timetable;
          it->second.beeline_ = beeline;
          it->second.beeline_oob_ = beeline_oob;
        } else {
          if (it->second.clasz_.has_value() && *it->second.clasz_ != clasz) {
            it->second.clasz_.reset();
          }
          it->second.beeline_ = it->second.beeline_ || beeline;
          it->second.beeline_oob_ = it->second.beeline_oob_ || beeline_oob;
        }
        for (auto const& short_name : short_names) {
          it->second.short_names_.insert(short_name);
        }
      };

  auto const n_routes = tt.route_location_seq_.size();
  progress_tracker->status("Collect Route Geometries")
      .out_bounds(1, 10)
      .in_high(std::max<std::size_t>(1U, n_routes));

  {
    auto const phase = scoped_timing{"collect_route_geometries"};
    for (auto route_idx = std::size_t{0U}; route_idx < n_routes; ++route_idx) {
      progress_tracker->update_monotonic(route_idx + 1U);

      auto const r = n::route_idx_t{route_idx};
      auto const route_clasz = std::string{to_str(tt.route_clasz_[r])};
      auto const& route_stops = tt.route_location_seq_[r];
      if (route_stops.size() < 2U || tt.route_transport_ranges_[r].empty()) {
        continue;
      }

      auto route_short_names = std::set<std::string>{};
      auto route_color_names = std::vector<std::string>{};
      auto route_color = std::optional<std::string>{};

      auto shape_added = false;
      for (auto const transport_idx : tt.route_transport_ranges_[r]) {
        auto const stop_indices = n::interval{
            n::stop_idx_t{0U}, static_cast<n::stop_idx_t>(route_stops.size())};

        for (auto const [from, to] : utl::pairwise(stop_indices)) {
          auto run = n::rt::run{};
          run.t_ = n::transport{transport_idx, n::day_idx_t{0}};
          run.stop_range_ =
              n::interval{from, static_cast<n::stop_idx_t>(to + 1U)};
          run.rt_ = n::rt_transport_idx_t::invalid();
          auto const fr = n::rt::frun{tt, nullptr, run};

          if (from == stop_indices.from_) {
            auto const short_name = std::string{
                fr[0].route_short_name(n::event_type::kDep, n::lang_t{})};
            add_unique(route_color_names, short_name);
            if (!short_name.empty()) {
              route_short_names.insert(short_name);
            }
            if (!route_color.has_value()) {
              if (auto const color =
                      to_str(fr[0].get_route_color(n::event_type::kDep).color_);
                  color.has_value()) {
                route_color = fmt::format("#{}", *color);
              }
            }
          }

          if (shape_added) {
            continue;
          }

          enc.reset();
          auto line = ::tiles::fixed_line{};
          fr.for_each_shape_point(
              shapes, n::interval{n::stop_idx_t{0U}, n::stop_idx_t{2U}},
              [&](auto const& p) {
                enc.push(p);
                line.push_back(::tiles::latlng_to_fixed({p.lat_, p.lng_}));
              });

          if (line.size() < 2U) {
            continue;
          }

          auto const from_location = fr[0].get_location_idx();
          auto const to_location = fr[1].get_location_idx();
          stop_locations.insert(from_location);
          stop_locations.insert(to_location);

          if (route_short_names.empty()) {
            auto const long_name = std::string{
                fr[0].route_long_name(n::event_type::kDep, n::lang_t{})};
            if (!long_name.empty()) {
              route_short_names.insert(long_name);
            } else {
              route_short_names.insert(fmt::format("route_{}", route_idx));
            }
          }
          auto const resolved_color =
              route_color.value_or(hash_color(join_comma(route_color_names)));
          auto const is_beeline = line.size() <= 2U;
          auto const beeline_oob =
              is_beeline &&
              is_beeline_oob(w, l, tt.locations_.coordinates_.at(from_location),
                             tt.locations_.coordinates_.at(to_location));
          add_route_segment(enc.buf_, std::move(line), route_short_names,
                            resolved_color, route_clasz,
                            route_color.has_value(), is_beeline, beeline_oob);
        }

        shape_added = true;
      }
    }
  }

  std::clog << "[route_tiles] collected routes=" << n_routes
            << ", unique_polylines=" << polyline_features.size()
            << ", stop_points=" << stop_locations.size() << "\n";

  auto beeline_segments = std::size_t{0U};
  auto beeline_oob_segments = std::size_t{0U};
  for (auto const& [_, route_feature] : polyline_features) {
    if (route_feature.beeline_) {
      ++beeline_segments;
    }
    if (route_feature.beeline_oob_) {
      ++beeline_oob_segments;
    }
  }
  std::clog << "[route_tiles] beeline_segments=" << beeline_segments << "\n";
  std::clog << "[route_tiles] beeline_oob_segments=" << beeline_oob_segments
            << "\n";

  auto feature_id = std::uint64_t{0U};

  progress_tracker->status("Insert Route Features")
      .out_bounds(10, 12)
      .in_high(std::max<std::size_t>(1U, polyline_features.size()));
  auto route_feature_idx = std::size_t{0U};
  {
    auto const phase = scoped_timing{"insert_route_features"};
    for (auto const& [_, route_feature] : polyline_features) {
      progress_tracker->update_monotonic(++route_feature_idx);

      auto f = ::tiles::feature{};
      f.id_ = feature_id++;
      f.layer_ = routes_layer_id;
      f.zoom_levels_ = {0U, ::tiles::kMaxZoomLevel};
      f.meta_.emplace_back(
          "route_short_names",
          ::tiles::encode_string(join_comma(route_feature.short_names_)));
      f.meta_.emplace_back("color",
                           ::tiles::encode_string(route_feature.color_));
      f.meta_.emplace_back(
          "clasz", ::tiles::encode_string(
                       route_feature.clasz_.value_or(std::string{"MIXED"})));
      f.meta_.emplace_back(
          "color_from_timetable",
          ::tiles::encode_bool(route_feature.color_from_timetable_));
      f.meta_.emplace_back("beeline",
                           ::tiles::encode_bool(route_feature.beeline_));
      f.meta_.emplace_back("beeline_oob",
                           ::tiles::encode_bool(route_feature.beeline_oob_));
      f.geometry_ = ::tiles::fixed_polyline{{route_feature.line_}};
      feature_inserter.insert(f);
    }
  }

  progress_tracker->status("Insert Stop Features")
      .out_bounds(12, 13)
      .in_high(std::max<std::size_t>(1U, stop_locations.size()));
  auto stop_feature_idx = std::size_t{0U};
  {
    auto const phase = scoped_timing{"insert_stop_features"};
    for (auto const l : stop_locations) {
      progress_tracker->update_monotonic(++stop_feature_idx);

      auto f = ::tiles::feature{};
      f.id_ = feature_id++;
      f.layer_ = stops_layer_id;
      f.zoom_levels_ = {9U, ::tiles::kMaxZoomLevel};
      f.meta_.emplace_back("id", ::tiles::encode_string(tags.id(tt, l)));
      auto const root = tt.locations_.get_root_idx(l);
      auto const name_loc = root == n::location_idx_t::invalid() ? l : root;
      f.meta_.emplace_back("name",
                           ::tiles::encode_string(tt.translate(
                               n::lang_t{}, tt.locations_.names_[name_loc])));
      auto const pos = tt.locations_.coordinates_.at(l);
      f.geometry_ = ::tiles::fixed_point{
          {::tiles::latlng_to_fixed({pos.lat_, pos.lng_})}};
      feature_inserter.insert(f);
    }
  }

  progress_tracker->status("Finalize Feature Packs").out_bounds(13, 14);
  {
    auto const phase = scoped_timing{"flush_feature_packs"};
    feature_inserter.flush(0U, 0U);
  }

  {
    auto const phase = scoped_timing{"store_layer_names"};
    auto txn = db_handle.make_txn();
    layer_names.store(db_handle, txn);
    txn.commit();
  }

  progress_tracker->status("Prepare Route Vector Tiles")
      .out_bounds(14, 99)
      .out_mod(0.01F);
  {
    auto const phase = scoped_timing{"prepare_tiles"};
    ::tiles::prepare_tiles(db_handle, pack_handle, 14);
  }

  progress_tracker->status("Load Route Tile Database").out_bounds(99, 100);
  {
    auto const phase = scoped_timing{"load_route_tile_database"};
    d.load_route_tiles();
  }

  auto const total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            clock_t::now() - total_start)
                            .count();
  std::clog << "[route_tiles] total: " << total_ms << " ms\n";
}

}  // namespace motis
