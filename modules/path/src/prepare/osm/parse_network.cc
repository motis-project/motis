#include "motis/path/prepare/osm/parse_network.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "geo/latlng.h"

#include "utl/erase_duplicates.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

#include "motis/hash_map.h"

#include "motis/core/common/logging.h"

#include "motis/path/prepare/osm/parser_commons.h"
#include "motis/path/prepare/osm_util.h"

namespace ml = motis::logging;

namespace motis::path {

template <typename Predicate>
std::vector<osm_way> parse_network(std::string const& osm_file,
                                   Predicate&& pred) {
  std::vector<raw_way> raw_ways;
  std::vector<std::unique_ptr<raw_node>> raw_node_mem;
  mcd::hash_map<int64_t, raw_node*> raw_nodes;

  ml::scoped_timer t_full("parse_network");

  {
    ml::scoped_timer t("parse_network|read ways");

    std::string yes = "yes";

    foreach_osm_way(osm_file, [&](auto&& way) {
      if (!pred(way)) {
        return;
      }
      raw_ways.emplace_back(
          way.id(), way.get_value_by_key("oneway", "") == yes,
          utl::to_vec(way.nodes(), [&](auto const& node_ref) -> raw_node* {
            auto it = raw_nodes.find(node_ref.ref());
            if (it == end(raw_nodes)) {
              raw_node_mem.emplace_back(
                  std::make_unique<raw_node>(node_ref.ref(), way.id()));
              raw_nodes[node_ref.ref()] = raw_node_mem.back().get();
              return raw_node_mem.back().get();
            } else {
              it->second->in_ways_.push_back(way.id());
              return it->second;
            }
          }));
    });
  }

  {
    ml::scoped_timer t("parse_network|read nodes");
    resolve_node_locations(osm_file, raw_nodes);
  }

  ml::scoped_timer t("parse_network|post processing");

  for (auto& node : raw_node_mem) {
    utl::erase_duplicates(node->in_ways_);
  }

  auto osm_ways = make_osm_ways(raw_ways);
  aggregate_osm_ways(osm_ways);
  return osm_ways;
}

std::vector<osm_way> parse_rail(std::string const& osm_file) {
  std::vector<std::string> included_railway{"rail", "light_rail",
                                            "narrow_gauge"};
  std::string yes{"yes"};
  std::vector<std::string> excluded_usages{"industrial", "military", "test",
                                           "tourism"};
  std::vector<std::string> excluded_services{"yard", "spur"};  // , "siding"

  return parse_network(osm_file, [&](auto&& way) {
    auto const railway = way.get_value_by_key("railway", "");
    if (std::none_of(begin(included_railway), end(included_railway),
                     [&](auto&& r) { return r == railway; })) {
      return false;
    }

    auto const usage = way.get_value_by_key("usage", "");
    if (std::any_of(begin(excluded_usages), end(excluded_usages),
                    [&](auto&& u) { return u == usage; })) {
      return false;
    }

    auto const service = way.get_value_by_key("service", "");
    if (std::any_of(begin(excluded_services), end(excluded_services),
                    [&](auto&& s) { return s == service; })) {
      return false;
    }

    if (yes == way.get_value_by_key("railway:preserved", "")) {
      return false;
    }
    return true;
  });
}

std::vector<osm_way> parse_subway(std::string const& osm_file) {
  std::vector<std::string> rail_route{"light_rail", "subway"};
  std::vector<std::string> excluded_usages{"industrial", "military", "test",
                                           "tourism"};

  return parse_network(osm_file, [&](auto&& way) {
    auto const rail = way.get_value_by_key("railway", "");
    if (std::none_of(begin(rail_route), end(rail_route),
                     [&](auto&& r) { return r == rail; })) {
      return false;
    }

    auto const usage = way.get_value_by_key("usage", "");
    if (std::any_of(begin(excluded_usages), end(excluded_usages),
                    [&](auto&& u) { return u == usage; })) {
      return false;
    }
    return true;
  });
}

std::vector<osm_way> parse_tram(std::string const& osm_file) {
  std::vector<std::string> routes{"tram"};

  return parse_network(osm_file, [&](auto&& way) {
    auto const rail = way.get_value_by_key("railway", "");
    if (std::none_of(begin(routes), end(routes),
                     [&](auto&& r) { return r == rail; })) {
      return false;
    }
    return true;
  });
}

std::vector<std::vector<osm_way>> parse_network(
    std::string const& osm_file, source_spec::category const& category) {
  switch (category) {
    case source_spec::category::RAIL: return {parse_rail(osm_file)};
    case source_spec::category::SUBWAY: return {parse_subway(osm_file)};
    case source_spec::category::TRAM: return {parse_tram(osm_file)};
    default: throw utl::fail("parse_network: unknown category");
  }
}

}  // namespace motis::path
