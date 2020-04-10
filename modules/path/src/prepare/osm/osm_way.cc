#include "motis/path/prepare/osm/osm_way.h"

#include <algorithm>

#include "motis/hash_map.h"

#include "utl/concat.h"
#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

namespace motis::path {

void aggregate_osm_ways(std::vector<osm_way>& osm_ways) {
  mcd::hash_map<int64_t, size_t> degrees;
  for (auto const& way : osm_ways) {
    utl::verify(way.is_valid(), "initially all ways must be valid!");
    degrees[way.from_] += 1;
    degrees[way.to_] += 1;
  }

  for (auto it = begin(osm_ways); it != end(osm_ways); ++it) {
    if (!it->is_valid() || it->from_ == it->to_) {
      continue;
    }
    while (degrees[it->from_] == 2) {
      if (it->from_ == it->to_) {
        break;  // cycle detected
      }

      auto other_it =
          std::find_if(std::next(it), end(osm_ways), [&](auto const& other) {
            return other.is_valid() &&
                   (it->from_ == other.from_ || it->from_ == other.to_);
          });

      if (other_it == end(osm_ways)) {
        break;
      }
      utl::verify(other_it != end(osm_ways), "osm_ways: node not found (from)");

      if (it->oneway_ != other_it->oneway_) {
        break;
      }

      if (it->from_ == other_it->to_) {
        //  --(other)--> X --(this)-->
        it->from_ = other_it->from_;
      } else {
        //  <--(other)-- X --(this)-->
        if (it->oneway_) {
          break;  // conflicting oneway directions
        }

        it->from_ = other_it->to_;

        other_it->path_.reverse();
      }

      other_it->path_.append(it->path_);
      it->path_ = std::move(other_it->path_);

      utl::concat(it->ids_, other_it->ids_);
      other_it->invalidate();
    }

    if (it->from_ == it->to_) {
      continue;  // cycle detected
    }

    while (degrees[it->to_] == 2) {
      if (it->from_ == it->to_) {
        break;  // cycle detected
      }

      auto other_it =
          std::find_if(std::next(it), end(osm_ways), [&](auto const& other) {
            return other.is_valid() &&
                   (it->to_ == other.from_ || it->to_ == other.to_);
          });

      if (other_it == end(osm_ways)) {
        break;
      }
      utl::verify(other_it != end(osm_ways), "osm_ways: node not found (to)");

      if (it->oneway_ != other_it->oneway_) {
        break;
      }

      if (it->to_ == other_it->from_) {
        // --(this)--> X --(other)-->
        it->to_ = other_it->to_;
      } else {
        // --(this)--> X <--(other)--
        if (it->oneway_) {
          break;  // conflicting oneway directions
        }

        it->to_ = other_it->from_;

        other_it->path_.reverse();
      }

      it->path_.append(other_it->path_);
      utl::concat(it->ids_, other_it->ids_);
      other_it->invalidate();
    }
  }

  utl::erase_if(osm_ways,
                [](auto const& osm_way) { return !osm_way.is_valid(); });
}

}  // namespace motis::path
