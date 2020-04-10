#pragma once

#include "osmium/handler.hpp"
#include "osmium/io/pbf_input.hpp"
#include "osmium/io/reader_iterator.hpp"
#include "osmium/memory/buffer.hpp"
#include "osmium/osm.hpp"
#include "osmium/visitor.hpp"

namespace motis::path {

template <typename F>
void foreach_osm_node(std::string const& filename, F f) {
  namespace oio = osmium::io;
  oio::Reader reader(filename, osmium::osm_entity_bits::node);
  for (auto it = oio::begin(reader); it != oio::end(reader); ++it) {
    f(static_cast<osmium::Node&>(*it));  // NOLINT
  }
}

template <typename F>
void foreach_osm_way(std::string const& filename, F f) {
  namespace oio = osmium::io;
  oio::Reader reader(filename, osmium::osm_entity_bits::way);
  for (auto it = oio::begin(reader); it != oio::end(reader); ++it) {
    f(static_cast<osmium::Way&>(*it));  // NOLINT
  }
}

template <typename F>
void foreach_osm_relation(std::string const& filename, F f) {
  namespace oio = osmium::io;
  oio::Reader reader(filename, osmium::osm_entity_bits::relation);
  for (auto it = oio::begin(reader); it != oio::end(reader); ++it) {
    f(static_cast<osmium::Relation&>(*it));  // NOLINT
  }
}

}  // namespace motis::path
