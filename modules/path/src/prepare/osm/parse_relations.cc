#include "motis/path/prepare/osm/parse_relations.h"

#include <cstdint>
#include <memory>

#include "geo/latlng.h"

#include "utl/erase_if.h"
#include "utl/get_or_create.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm/parser_commons.h"
#include "motis/path/prepare/osm_util.h"

namespace ml = motis::logging;

namespace motis::path {

std::vector<std::vector<osm_way>> parse_relations(
    std::string const& osm_file, source_spec::category const& category) {
  ml::scoped_timer t_full("parse_relations");

  std::vector<std::vector<raw_way*>> relations;
  std::vector<std::unique_ptr<raw_way>> way_mem;
  std::vector<std::unique_ptr<raw_node>> node_mem;

  mcd::hash_map<int64_t, raw_way*> pending_ways;
  mcd::hash_map<int64_t, raw_node*> pending_nodes;

  {
    ml::scoped_timer t("parse_relations|read relations");

    std::vector<std::string> types{"route", "public_transport"};
    std::vector<std::string> rail_routes{"railway", "train"};
    std::vector<std::string> tram_routes{"tram"};
    std::vector<std::string> bus_routes{"bus"};
    std::vector<std::string> sub_routes{"light_rail", "subway"};

    foreach_osm_relation(osm_file, [&](auto&& relation) {
      auto const type = relation.get_value_by_key("type", "");
      if (std::none_of(begin(types), end(types),
                       [&](auto&& t) { return t == type; })) {
        return;
      }

      auto const route = relation.get_value_by_key("route", "");

      auto const is_rail = std::any_of(begin(rail_routes), end(rail_routes),
                                       [&](auto&& r) { return r == route; });
      auto const is_bus = std::any_of(begin(bus_routes), end(bus_routes),
                                      [&](auto&& r) { return r == route; });
      auto const is_sub = std::any_of(begin(sub_routes), end(sub_routes),
                                      [&](auto&& r) { return r == route; });
      auto const is_tram = std::any_of(begin(tram_routes), end(tram_routes),
                                       [&](auto&& r) { return r == route; });
      if (!is_rail && !is_bus && !is_sub && !is_tram) {
        return;
      }

      auto cat =
          is_rail ? source_spec::category::RAILWAY
                  : is_bus ? source_spec::category::BUS
                           : is_sub ? source_spec::category::SUBWAY
                                    : is_tram ? source_spec::category::TRAM
                                              : source_spec::category::UNKNOWN;
      if (category != source_spec::category::UNKNOWN && category != cat) {
        return;
      }

      std::vector<raw_way*> ways;
      for (auto const& member : relation.members()) {
        if (member.type() != osmium::item_type::way) {
          continue;
        }

        ways.push_back(utl::get_or_create(pending_ways, member.ref(), [&] {
          way_mem.push_back(std::make_unique<raw_way>(member.ref()));
          return way_mem.back().get();
        }));
      }

      relations.emplace_back(std::move(ways));
    });
  }

  {
    ml::scoped_timer t("parse_relations|read ways");

    std::string platform = "platform";
    std::string stop = "stop";
    std::string yes = "yes";

    foreach_osm_way(osm_file, [&](auto&& way) {
      auto w = pending_ways.find(way.id());
      if (w == end(pending_ways) ||
          platform == way.get_value_by_key("highway", "") ||
          platform == way.get_value_by_key("public_transport", "") ||
          stop == way.get_value_by_key("role", "") ||
          way.tags().has_key("building")) {
        return;
      }

      w->second->oneway_ = way.get_value_by_key("oneway", "") == yes;
      w->second->resolved_ = true;

      for (auto const& node : way.nodes()) {
        auto const it = pending_nodes.find(node.ref());
        if (it == end(pending_nodes)) {
          node_mem.emplace_back(
              std::make_unique<raw_node>(node.ref(), way.id()));
          pending_nodes[node.ref()] = node_mem.back().get();
          w->second->nodes_.push_back(node_mem.back().get());
        } else {
          it->second->in_ways_.push_back(way.id());
          w->second->nodes_.push_back(it->second);
        }
      }
    });
  }

  {
    ml::scoped_timer t("parse_relations|read nodes");
    thread_pool tp;
    resolve_node_locations(osm_file, pending_nodes, tp);
  }

  ml::scoped_timer t("parse_relations|post processing");
  for (auto& rel : relations) {
    for (auto& way : rel) {
      for (auto& node : way->nodes_) {
        if (!node->resolved_) {
          std::clog << "missing node" << node->id_ << std::endl;
        }
      }
    }
    utl::erase_if(rel, [&](auto const& way) {
      return !way->resolved_ || way->nodes_.size() < 2;
    });
  }

  utl::verify(!relations.empty(), "no relations found");
  LOG(motis::logging::info) << "found " << relations.size() << " relations";

  return utl::to_vec(relations, [](auto const& way_ptrs) {
    auto osm_ways = make_osm_ways(
        utl::to_vec(way_ptrs, [](auto const& ptr) { return *ptr; }));
    aggregate_osm_ways(osm_ways);
    return osm_ways;
  });
}

}  // namespace motis::path
