#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/geometry.hpp"

#include "motis/intermodal/eval/parse_poly.h"

namespace bg = boost::geometry;

namespace motis::intermodal::eval {

namespace {
enum class state { NAME, MULTIPOLY, OUTER_RING, INNER_RING };
}  // namespace

std::unique_ptr<poly> parse_poly(std::string const& filename) {
  std::ifstream f(filename);
  if (!f) {
    std::cerr << "Could not read poly file: " << filename << std::endl;
    return nullptr;
  }

  poly::multi_polygon_t multipoly;
  std::vector<geo::latlng> ring;

  state st = state::NAME;

  for (std::string line; std::getline(f, line);) {
    switch (st) {
      case state::NAME: st = state::MULTIPOLY; break;

      case state::MULTIPOLY:
        if (!line.empty() && !boost::starts_with(line, "END")) {
          st = line[0] == '!' ? state::INNER_RING : state::OUTER_RING;
        }
        break;

      case state::OUTER_RING:
      case state::INNER_RING:
        if (boost::starts_with(line, "END")) {
          if (ring.empty()) {
            std::cerr << "Invalid poly file: empty ring: " << filename
                      << std::endl;
            return nullptr;
          }
          if (!(ring.back() == ring.front())) {
            ring.push_back(ring.front());
          }
          if (st == state::OUTER_RING) {
            multipoly.resize(multipoly.size() + 1);
            for (auto const& p : ring) {
              bg::append(multipoly.back(), p);
            }
          } else {
            multipoly.back().inners().resize(multipoly.back().inners().size() +
                                             1);
            for (auto const& p : ring) {
              bg::append(multipoly.back().inners().back(), p);
            }
          }
          st = state::MULTIPOLY;
          ring.clear();
        } else {
          std::istringstream ss(line);
          double lon = NAN, lat = NAN;
          if (!(ss >> lon >> lat)) {
            std::cerr << "Syntax error in poly file: " << filename << ": "
                      << line << std::endl;
            return nullptr;
          }
          ring.emplace_back(geo::latlng{lat, lon});
        }
        break;
    }
  }

  return std::make_unique<poly>(multipoly);
}

}  // namespace motis::intermodal::eval
