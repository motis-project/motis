#include "motis/path/prepare/osm/osm_data.h"

#include "boost/filesystem.hpp"

#include "cista/serialization.h"

#include "osmium/area/assembler.hpp"
#include "osmium/area/multipolygon_manager.hpp"
#include "osmium/io/pbf_input.hpp"
#include "osmium/io/reader.hpp"
#include "osmium/memory/buffer.hpp"
#include "osmium/visitor.hpp"

#include "utl/equal_ranges_linear.h"
#include "utl/erase_duplicates.h"
#include "utl/get_or_create.h"
#include "utl/pipes.h"
#include "utl/progress_tracker.h"
#include "utl/to_vec.h"
#include "utl/verify.h"
#include "utl/visit.h"
#include "utl/zip.h"

#include "tiles/osm/hybrid_node_idx.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/cista_util.h"
#include "motis/path/prepare/osm/osm_way.h"

namespace o = osmium;
namespace oio = osmium::io;
namespace oh = osmium::handler;
namespace oeb = osmium::osm_entity_bits;

namespace motis::path {

constexpr auto const CISTA_MODE =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_VERSION;

auto const str_ferry = std::string_view{"ferry"};
auto const str_platform = std::string_view{"platform"};
auto const str_stop_position = std::string_view{"stop_position"};
auto const str_stop = std::string_view{"stop"};
auto const str_yes = std::string_view{"yes"};

struct raw_way {
  explicit raw_way(o::object_id_type id) : id_{id}, oneway_{false} {}

  raw_way(o::object_id_type id, bool oneway,
          std::vector<o::object_id_type> node_ids)
      : id_{id}, oneway_{oneway}, node_ids_{std::move(node_ids)} {}

  osm_way extract(size_t const b, size_t const e) const {
    utl::verify(b != e, "extract empty way!");
    return osm_way{
        {id_},
        oneway_,
        osm_path{
            mcd::to_vec(begin(locations_) + b, begin(locations_) + e,
                        [](auto const& l) {
                          return geo::latlng{l.lat(), l.lon()};
                        }),
            mcd::vector<int64_t>{begin(node_ids_) + b, begin(node_ids_) + e}}};
  }

  o::object_id_type id_;
  bool oneway_;
  std::vector<o::object_id_type> node_ids_;
  std::vector<o::Location> locations_;
};

mcd::vector<osm_way> make_osm_ways(std::vector<raw_way> const& raw_ways) {
  std::vector<o::object_id_type> in_multiple_ways;
  {
    std::vector<o::object_id_type> node_ids;
    for (auto const& raw_way : raw_ways) {
      utl::concat(node_ids, raw_way.node_ids_);
    }
    std::sort(begin(node_ids), end(node_ids));
    utl::equal_ranges_linear(node_ids, [&](auto lb, auto ub) {
      if (std::distance(lb, ub) > 1) {
        in_multiple_ways.push_back(*lb);
      }
    });
  }

  mcd::vector<osm_way> osm_ways;
  for (auto const& way : raw_ways) {
    if (way.node_ids_.size() < 2 || way.locations_.empty()) {
      continue;
    }
    utl::verify(way.node_ids_.size() == way.locations_.size(),
                "way not resolved!");

    auto from = begin(way.node_ids_);
    while (true) {
      auto to =
          std::find_if(std::next(from), end(way.node_ids_), [&](auto const id) {
            auto const it = std::lower_bound(begin(in_multiple_ways),
                                             end(in_multiple_ways), id);
            return it != end(in_multiple_ways) && *it == id;
          });

      if (to == end(way.node_ids_)) {
        break;
      }

      osm_ways.push_back(
          way.extract(std::distance(begin(way.node_ids_), from),
                      std::distance(begin(way.node_ids_), to) + 1));
      from = to;
    }
    if (std::distance(from, end(way.node_ids_)) >= 2) {
      osm_ways.push_back(way.extract(std::distance(begin(way.node_ids_), from),
                                     way.node_ids_.size()));
    }
  }
  return osm_ways;
}

struct relation_way_base {
  void add_relation(o::Relation const& r) {
    auto vec =
        utl::all(r.members())  //
        | utl::remove_if(
              [](auto&& m) { return m.type() != o::item_type::way; })  //
        |
        utl::transform([&](auto&& m) {
          return utl::get_or_create(ways_, m.ref(), [&] {
            return mem_.emplace_back(std::make_unique<raw_way>(m.ref())).get();
          });
        })  //
        | utl::vec();

    if (!vec.empty()) {
      relations_.emplace_back(std::move(vec));
    }
  }

  bool resolve_way(o::Way const& w) {
    auto it = ways_.find(w.id());
    if (it == end(ways_)) {
      return false;
    }

    it->second->oneway_ = str_yes == w.get_value_by_key("oneway", "");
    it->second->node_ids_ =
        utl::to_vec(w.nodes(), [](auto const& n) { return n.ref(); });
    return true;
  }

  void add_relation_way(o::Way const& w) {
    raw_way* rw = utl::get_or_create(ways_, w.id(), [&] {
      return mem_.emplace_back(std::make_unique<raw_way>(w.id())).get();
    });

    rw->oneway_ = str_yes == w.get_value_by_key("oneway", "");
    rw->node_ids_ =
        utl::to_vec(w.nodes(), [](auto const& n) { return n.ref(); });

    relations_.push_back(std::vector<raw_way*>{rw});
  }

  std::vector<std::vector<raw_way*>> relations_;
  mcd::hash_map<o::object_id_type, raw_way*> ways_;
  std::vector<std::unique_ptr<raw_way>> mem_;
};

struct relation_handler : public oh::Handler, relation_way_base {
  explicit relation_handler(std::vector<std::string> allowed_routes)
      : allowed_routes_{std::move(allowed_routes)} {}

  void relation(o::Relation const& r) {
    auto const* type = r.get_value_by_key("type", "");
    auto const* route = r.get_value_by_key("route", "");
    if (std::none_of(begin(allowed_types_), end(allowed_types_),
                     [&](auto&& t) { return t == type; }) ||
        std::none_of(begin(allowed_routes_), end(allowed_routes_),
                     [&](auto&& r) { return r == route; })) {
      return;
    }

    add_relation(r);
  }

  void way(o::Way const& w) {
    if (str_platform == w.get_value_by_key("highway", "") ||
        str_platform == w.get_value_by_key("public_transport", "") ||
        str_stop == w.get_value_by_key("role", "") ||
        w.tags().has_key("building")) {
      return;
    }
    resolve_way(w);
  }

  mcd::vector<mcd::vector<osm_way>> finalize() {
    mcd::vector<mcd::vector<osm_way>> result;
    for (auto const& way_ptrs : relations_) {
      auto osm_ways = make_osm_ways(
          utl::to_vec(way_ptrs, [](auto const& ptr) { return *ptr; }));
      if (!osm_ways.empty()) {
        result.emplace_back(aggregate_osm_ways(std::move(osm_ways)));
      }
    }
    return result;
  }

  std::vector<std::string> allowed_routes_;
  std::vector<std::string> allowed_types_{"route", "public_transport"};
};

template <typename Fn>
struct network_handler : public oh::Handler {
  explicit network_handler(Fn fn) : fn_{std::move(fn)} {}

  void way(o::Way const& w) {
    if (fn_(w)) {
      ways_.emplace_back(
          w.id(), str_yes == w.get_value_by_key("oneway", ""),
          utl::to_vec(w.nodes(), [](auto const& n) { return n.ref(); }));
    }
  }

  mcd::vector<mcd::vector<osm_way>> finalize() {
    return {aggregate_osm_ways(make_osm_ways(ways_))};
  }

  Fn fn_;
  std::vector<raw_way> ways_;
};

bool match_any(o::Way const& way, char const* key,
               std::vector<std::string> const& values) {
  auto const v = way.get_value_by_key(key, "");
  return std::any_of(begin(values), end(values),
                     [&](auto const& value) { return value == v; });
}

bool match_none(o::Way const& way, char const* key,
                std::vector<std::string> const& values) {
  auto const v = way.get_value_by_key(key, "");
  return std::none_of(begin(values), end(values),
                      [&](auto const& value) { return value == v; });
}

struct stop_position_handler : public oh::Handler {
  void node(o::Node const& n) {
    if (str_stop_position != n.get_value_by_key("public_transport", "") ||
        str_yes != n.get_value_by_key("bus", "")) {
      return;
    }
    coordinates_.emplace_back(n.location().lat(), n.location().lon());
  }

  mcd::vector<geo::latlng> coordinates_;
};

struct plattform_handler : public oh::Handler, relation_way_base {
  void relation(o::Relation const& r) {
    auto const* public_transport = r.get_value_by_key("public_transport", "");
    if (str_platform != public_transport) {
      return;
    }

    add_relation(r);
  }

  void way(o::Way const& w) {
    if (resolve_way(w)) {
      return;
    }

    auto const* public_transport = w.get_value_by_key("public_transport", "");
    if (str_platform == public_transport) {
      add_relation_way(w);
    }
  }

  mcd::vector<geo::latlng> finalize() const {
    mcd::vector<geo::latlng> result;
    for (auto const& ways : relations_) {
      std::vector<std::pair<o::object_id_type, o::Location>> coords;
      for (auto const& w : ways) {
        if (w->locations_.empty()) {
          continue;
        }
        for (auto const& [id, l] : utl::zip(w->node_ids_, w->locations_)) {
          coords.emplace_back(id, l);
        }
      }

      if (coords.empty()) {
        continue;
      }

      utl::erase_duplicates(
          coords,
          [](auto const& a, auto const& b) { return a.first < b.first; },
          [](auto const& a, auto const& b) { return a.first == b.first; });

      double lat_sum{0};
      double lng_sum{0};
      for (auto const& [id, loc] : coords) {
        lat_sum += loc.lat();
        lng_sum += loc.lon();
      }
      result.emplace_back(lat_sum / coords.size(), lng_sum / coords.size());
    }
    return result;
  }
};

mcd::unique_ptr<osm_data> parse_osm(std::string const& osm_file) {
  logging::scoped_timer timer("parse_osm");
  auto progress_tracker = utl::get_active_progress_tracker();

  auto rel_rail = relation_handler{{"railway", "train"}};
  auto rel_sub = relation_handler{{"light_rail", "subway"}};
  auto rel_tram = relation_handler{{"tram"}};
  auto rel_bus = relation_handler{{"bus"}};

  std::vector<std::string> rail_incl{"rail", "light_rail", "narrow_gauge"};
  std::vector<std::string> rail_excl_usage{"industrial", "military", "test"};
  std::vector<std::string> rail_excl_service{"yard", "spur"};
  auto net_rail = network_handler{[&](auto const& w) {
    return match_any(w, "railway", rail_incl) &&
           match_none(w, "usage", rail_excl_usage) &&
           match_none(w, "service", rail_excl_service) &&
           str_yes != w.get_value_by_key("railway:preserved", "");
  }};

  std::vector<std::string> subway_incl{"light_rail", "subway"};
  std::vector<std::string> subway_excl_usage{"industrial", "military", "test",
                                             "tourism"};
  auto net_sub = network_handler{[&](auto const& w) {
    return match_any(w, "railway", subway_incl) &&
           match_none(w, "usage", subway_excl_usage);
  }};

  std::vector<std::string> tram_incl{"tram"};
  auto net_tram = network_handler{
      [&](auto const& w) { return match_any(w, "railway", tram_incl); }};

  std::vector<std::string> waterway_incl{"river", "canal"};
  auto net_ship = network_handler{[&](auto const& w) {
    return str_ferry == w.get_value_by_key("route", "") ||
           match_any(w, "waterway", waterway_incl);
  }};

  auto stop_positions = stop_position_handler{};
  auto plattforms = plattform_handler{};

  {
    oio::File input_file{osm_file};
    tiles::hybrid_node_idx node_idx;

    {
      tiles::hybrid_node_idx_builder node_idx_builder{node_idx};

      oio::Reader reader{input_file, oeb::node | oeb::relation};
      progress_tracker->status("Load OSM / Pass 1")
          .out_bounds(5, 15)
          .in_high(reader.file_size());
      while (auto buffer = reader.read()) {
        progress_tracker->update(reader.offset());
        o::apply(buffer, node_idx_builder, stop_positions, plattforms,  //
                 rel_rail, rel_tram, rel_bus, rel_sub);
      }
      reader.close();

      node_idx_builder.finish();
    }
    {
      oio::Reader reader{input_file, oeb::way};
      progress_tracker->status("Load OSM / Pass 2")
          .out_bounds(15, 25)
          .in_high(reader.file_size());
      while (auto buffer = reader.read()) {
        progress_tracker->update(reader.offset());
        o::apply(buffer, plattforms,  //
                 rel_rail, rel_sub, rel_tram, rel_bus,  //
                 net_rail, net_sub, net_tram, net_ship);
      }
    }

    progress_tracker->status("Load OSM / Locations");
    std::vector<std::pair<o::object_id_type, o::Location*>> locations;
    auto const collect_ways = [&](raw_way& w) {
      w.locations_.resize(w.node_ids_.size());
      for (auto i = 0ULL; i < w.node_ids_.size(); ++i) {
        locations.emplace_back(w.node_ids_[i], &w.locations_[i]);
      }
    };
    auto const collect =
        overloaded{[&](raw_way& w) { collect_ways(w); },
                   [&](std::unique_ptr<raw_way>& w) { collect_ways(*w); }};

    std::for_each(begin(plattforms.mem_), end(plattforms.mem_), collect);

    std::for_each(begin(rel_rail.mem_), end(rel_rail.mem_), collect);
    std::for_each(begin(rel_sub.mem_), end(rel_sub.mem_), collect);
    std::for_each(begin(rel_tram.mem_), end(rel_tram.mem_), collect);
    std::for_each(begin(rel_bus.mem_), end(rel_bus.mem_), collect);

    std::for_each(begin(net_rail.ways_), end(net_rail.ways_), collect);
    std::for_each(begin(net_sub.ways_), end(net_sub.ways_), collect);
    std::for_each(begin(net_tram.ways_), end(net_tram.ways_), collect);
    std::for_each(begin(net_ship.ways_), end(net_ship.ways_), collect);

    std::sort(begin(locations), end(locations));

    get_coords(node_idx, locations);
    for (auto const& [id, l] : locations) {
      l->set_x(l->x() - tiles::hybrid_node_idx::x_offset);
      l->set_y(l->y() - tiles::hybrid_node_idx::y_offset);
    }
  }

  progress_tracker->status("Load OSM / Finalize");
  using category = source_spec::category;
  using router = source_spec::router;
  auto data = mcd::make_unique<osm_data>();
  data->stop_positions_ = std::move(stop_positions.coordinates_);
  data->plattforms_ = plattforms.finalize();

  auto const finalize = [&](source_spec const ss, auto& handler) {
    progress_tracker->status(fmt::format("Load OSM / Finalize {}", ss.str()));
    data->profiles_[ss] = handler.finalize();
  };

  finalize({category::RAIL, router::OSM_REL}, rel_rail);
  finalize({category::SUBWAY, router::OSM_REL}, rel_sub);
  finalize({category::TRAM, router::OSM_REL}, rel_tram);
  finalize({category::BUS, router::OSM_REL}, rel_bus);
  finalize({category::RAIL, router::OSM_NET}, net_rail);
  finalize({category::SUBWAY, router::OSM_NET}, net_sub);
  finalize({category::TRAM, router::OSM_NET}, net_tram);
  finalize({category::SHIP, router::OSM_NET}, net_ship);
  return data;
}

mcd::unique_ptr<osm_data> read_osm_data(std::string const& fname,
                                        cista::memory_holder& mem) {
  mcd::unique_ptr<osm_data> ptr;
  ptr.self_allocated_ = false;
#if defined(MOTIS_SCHEDULE_MODE_OFFSET) && !defined(CLANG_TIDY)
  mem = cista::buf<cista::mmap>(
      cista::mmap{fname.c_str(), cista::mmap::protection::READ});
  ptr.el_ = cista::deserialize<osm_data, CISTA_MODE>(
      std::get<cista::buf<cista::mmap>>(mem));
#elif defined(MOTIS_SCHEDULE_MODE_RAW) || defined(CLANG_TIDY)
  mem = cista::file(fname.c_str(), "r").content();
  ptr.el_ =
      cista::deserialize<osm_data, CISTA_MODE>(std::get<cista::buffer>(mem));
#else
#error "no ptr mode specified"
#endif
  return ptr;
}

void write_osm_data(std::string const& fname,
                    mcd::unique_ptr<osm_data> const& data) {
  auto writer = cista::buf<cista::mmap>(
      cista::mmap{fname.c_str(), cista::mmap::protection::WRITE});
  cista::serialize<CISTA_MODE>(writer, *data);
}

}  // namespace motis::path
