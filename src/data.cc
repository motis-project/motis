#include "motis/data.h"

#include <filesystem>
#include <future>

#include "cista/io.h"

#include "utl/read_file.h"
#include "utl/verify.h"

#include "adr/adr.h"
#include "adr/cache.h"
#include "adr/reverse.h"
#include "adr/typeahead.h"

#include "osr/elevation_storage.h"
#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/ways.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"

#include "motis/config.h"
#include "motis/constants.h"
#include "motis/elevators/update_elevators.h"
#include "motis/hashes.h"
#include "motis/match_platforms.h"
#include "motis/odm/bounds.h"
#include "motis/point_rtree.h"
#include "motis/railviz.h"
#include "motis/tag_lookup.h"
#include "motis/tiles_data.h"
#include "motis/tt_location_rtree.h"

namespace fs = std::filesystem;
namespace n = nigiri;

namespace motis {

rt::rt() = default;

rt::rt(ptr<nigiri::rt_timetable>&& rtt,
       ptr<elevators>&& e,
       ptr<railviz_rt_index>&& railviz)
    : rtt_{std::move(rtt)}, railviz_rt_{std::move(railviz)}, e_{std::move(e)} {}

rt::~rt() = default;

std::ostream& operator<<(std::ostream& out, data const& d) {
  return out << "\nt=" << d.t_.get() << "\nr=" << d.r_ << "\ntc=" << d.tc_
             << "\nw=" << d.w_ << "\npl=" << d.pl_ << "\nl=" << d.l_
             << "\ntt=" << d.tt_.get()
             << "\nlocation_rtee=" << d.location_rtree_
             << "\nelevator_nodes=" << d.elevator_nodes_
             << "\nmatches=" << d.matches_ << "\nrt=" << d.rt_ << "\n";
}

data::data(std::filesystem::path p)
    : path_{std::move(p)},
      config_{config::read(path_ / "config.yml")},
      metrics_{std::make_unique<prometheus::Registry>()} {}

data::data(std::filesystem::path p, config const& c)
    : path_{std::move(p)}, config_{c} {
  auto const verify_version = [&](bool cond, char const* name, auto&& ver) {
    if (!cond) {
      return;
    }
    auto const [key, expected_ver] = ver;
    auto const h = read_hashes(path_, name);
    auto const existing_ver_it = h.find(key);

    utl::verify(existing_ver_it != end(h),
                "{}: no existing version found [key={}], please re-run import; "
                "hashes: {}",
                name, key, to_str(h));
    utl::verify(existing_ver_it->second == expected_ver,
                "{}: binary version mismatch [existing={} vs expected={}], "
                "please re-run import; hashes: {}",
                name, existing_ver_it->second, expected_ver, to_str(h));
  };

  verify_version(c.timetable_.has_value(), "tt", n_version());
  verify_version(c.geocoding_ || c.reverse_geocoding_, "adr", adr_version());
  verify_version(c.use_street_routing(), "osr", osr_version());
  verify_version(c.use_street_routing() && c.timetable_, "matches",
                 matches_version());
  verify_version(c.tiles_.has_value(), "tiles", tiles_version());
  verify_version(c.osr_footpath_, "osr_footpath", osr_footpath_version());

  rt_ = std::make_shared<rt>();

  if (c.odm_.has_value() && c.odm_->bounds_.has_value()) {
    odm_bounds_ = std::make_unique<odm::bounds>(*c.odm_->bounds_);
  }

  auto geocoder = std::async(std::launch::async, [&]() {
    if (c.geocoding_) {
      load_geocoder();
    }
    if (c.reverse_geocoding_) {
      load_reverse_geocoder();
    }
  });

  auto tt = std::async(std::launch::async, [&]() {
    if (c.timetable_) {
      load_tt(config_.osr_footpath_ ? "tt_ext.bin" : "tt.bin");
      if (c.timetable_->with_shapes_) {
        load_shapes();
      }
      if (c.timetable_->railviz_) {
        load_railviz();
      }
    }
  });

  auto street_routing = std::async(std::launch::async, [&]() {
    if (c.use_street_routing()) {
      load_osr();
    }
  });

  auto matches = std::async(std::launch::async, [&]() {
    if (c.use_street_routing() && c.timetable_) {
      load_matches();
    }
  });

  auto elevators = std::async(std::launch::async, [&]() {
    if (c.has_elevators()) {
      street_routing.wait();
      rt_->e_ = std::make_unique<motis::elevators>(
          *w_, *elevator_nodes_, vector_map<elevator_idx_t, elevator>{});

      if (c.get_elevators()->init_) {
        tt.wait();
        auto new_rtt = std::make_unique<n::rt_timetable>(
            n::rt::create_rt_timetable(*tt_, rt_->rtt_->base_day_));
        rt_->e_ =
            update_elevators(c, *this,
                             cista::mmap{c.get_elevators()->init_->c_str(),
                                         cista::mmap::protection::READ}
                                 .view(),
                             *new_rtt);
        rt_->rtt_ = std::move(new_rtt);
      }
    }
  });

  auto tiles = std::async(std::launch::async, [&]() {
    if (c.tiles_) {
      load_tiles();
    }
  });

  auto const throw_if_failed = [](char const* context, auto& future) {
    try {
      future.get();
    } catch (std::exception const& e) {
      throw utl::fail(
          "loading {} failed (if this happens after a fresh import, please "
          "file a bug report): {}",
          context, e.what());
    }
  };

  geocoder.wait();
  tt.wait();
  street_routing.wait();
  matches.wait();
  elevators.wait();
  tiles.wait();

  throw_if_failed("geocoder", geocoder);
  throw_if_failed("tt", tt);
  throw_if_failed("street_routing", street_routing);
  throw_if_failed("matches", matches);
  throw_if_failed("elevators", elevators);
  throw_if_failed("tiles", tiles);

  utl_verify(
      shapes_ == nullptr || tt_ == nullptr ||
          (tt_->n_routes() == shapes_->route_bboxes_.size() &&
           tt_->n_routes() == shapes_->route_segment_bboxes_.size()),
      "mismatch: n_routes={}, n_route_bboxes={}, n_route_segment_bboxes={}",
      tt_->n_routes(), shapes_->route_bboxes_.size(),
      shapes_->route_segment_bboxes_.size());
  utl_verify(matches_ == nullptr || tt_ == nullptr ||
                 matches_->size() == tt_->n_locations(),
             "mismatch: n_matches={}, n_locations={}", matches_->size(),
             tt_->n_locations());
}

data::~data() = default;
data::data(data&&) = default;
data& data::operator=(data&&) = default;

void data::load_osr() {
  auto const osr_path = path_ / "osr";
  w_ = std::make_unique<osr::ways>(osr_path, cista::mmap::protection::READ);
  l_ = std::make_unique<osr::lookup>(*w_, osr_path,
                                     cista::mmap::protection::READ);
  if (config_.get_street_routing()->elevation_data_dir_.has_value()) {
    elevations_ = osr::elevation_storage::try_open(osr_path);
  }
  elevator_nodes_ =
      std::make_unique<hash_set<osr::node_idx_t>>(get_elevator_nodes(*w_));
  pl_ =
      std::make_unique<osr::platforms>(osr_path, cista::mmap::protection::READ);
  pl_->build_rtree(*w_);
}

void data::load_tt(fs::path const& p) {
  tags_ = tag_lookup::read(path_ / "tags.bin");
  tt_ = n::timetable::read(path_ / p);
  tt_->resolve();
  location_rtree_ = std::make_unique<point_rtree<n::location_idx_t>>(
      create_location_rtree(*tt_));
  init_rtt();
}

void data::init_rtt(date::sys_days const d) {
  rt_->rtt_ =
      std::make_unique<n::rt_timetable>(n::rt::create_rt_timetable(*tt_, d));
}

void data::load_shapes() {
  shapes_ = {};
  shapes_ = std::make_unique<nigiri::shapes_storage>(
      nigiri::shapes_storage{path_, cista::mmap::protection::READ});
}

void data::load_railviz() {
  railviz_static_ = std::make_unique<railviz_static_index>(*tt_, shapes_.get());
  rt_->railviz_rt_ = std::make_unique<railviz_rt_index>(*tt_, *rt_->rtt_);
}

void data::load_geocoder() {
  t_ = adr::read(path_ / "adr" /
                 (config_.timetable_.has_value() ? "t_ext.bin" : "t.bin"));
  tc_ = std::make_unique<adr::cache>(t_->strings_.size(), 100U);
}

void data::load_reverse_geocoder() {
  r_ = std::make_unique<adr::reverse>(path_ / "adr",
                                      cista::mmap::protection::READ);
}

void data::load_matches() {
  matches_ = cista::read<platform_matches_t>(path_ / "matches.bin");
}

void data::load_tiles() {
  auto const db_size = config_.tiles_.value().db_size_;
  tiles_ = std::make_unique<tiles_data>(
      (path_ / "tiles" / "tiles.mdb").generic_string(), db_size);
}

}  // namespace motis
