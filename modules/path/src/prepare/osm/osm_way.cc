#include "motis/path/prepare/osm/osm_way.h"

#include <algorithm>
#include <memory>
#include <stack>

#include "motis/hash_map.h"

#include "utl/concat.h"
#include "utl/erase_duplicates.h"
#include "utl/erase_if.h"
#include "utl/to_vec.h"
#include "utl/verify.h"

namespace motis::path {

struct way;
using way_handle = std::unique_ptr<way>;

struct way {
  int64_t from_{}, to_{};

  way_handle left_, right_;
  osm_way* geometry_{nullptr};

  bool reversed_{false};
  bool oneway_{false};
};

constexpr auto const kInvalidId = std::numeric_limits<int64_t>::max();

std::vector<way_handle> make_way_handles(mcd::vector<osm_way>& ways) {
  std::vector<way_handle> way_handles;
  way_handles.reserve(ways.size());
  for (auto& w : ways) {
    auto wh = std::make_unique<way>();
    wh->from_ = w.from();
    wh->to_ = w.to();
    wh->geometry_ = &w;
    wh->oneway_ = w.oneway_;
    way_handles.emplace_back(std::move(wh));
  }
  return way_handles;
}

void join_ways(std::vector<way_handle>& handles) {
  std::vector<std::pair<int64_t, way_handle*>> idx;
  for (auto& lh : handles) {
    idx.emplace_back(lh->from_, &lh);
    idx.emplace_back(lh->to_, &lh);
  }
  utl::erase_duplicates(idx);

  auto const find_incident_line = [&](way_handle const& lh,
                                      int64_t const id) -> way_handle* {
    if (id == kInvalidId) {
      return nullptr;
    }

    auto it0 = std::lower_bound(
        begin(idx), end(idx), id,
        [](auto const& a, auto const& b) { return a.first < b; });

    size_t count = 0;
    way_handle* other = nullptr;
    for (auto it = it0; it != end(idx) && it->first == id; ++it) {
      ++count;
      // NOLINTNEXTLINE
      if (it->second->get() == lh.get() || it->second->get() == nullptr) {
        continue;  // found self or already gone
      }
      other = it->second;
    }

    if (count == 2) {
      return other;
    }

    // degree != 2 -> "burn" this coordinate for further processing
    for (auto it = it0; it != end(idx) && it->first == id; ++it) {
      // NOLINTNEXTLINE
      if (it->second->get() == nullptr) {
        continue;  // self can already be gone in bwd pass
      }

      if ((**it->second).from_ == id) {
        (**it->second).from_ = kInvalidId;
      }
      if ((**it->second).to_ == id) {
        (**it->second).to_ = kInvalidId;
      }
    }

    return nullptr;
  };

  auto const mark_reversed = [](way* w) {
    std::stack<way*> stack{{w}};
    while (!stack.empty()) {
      auto* curr = stack.top();
      stack.pop();

      curr->reversed_ = !curr->reversed_;

      if (curr->left_) {
        stack.push(curr->left_.get());
      }
      if (curr->right_) {
        stack.push(curr->right_.get());
      }
    }
  };

  for (auto it = begin(handles); it != end(handles); ++it) {
    if (!*it || (**it).from_ == (**it).to_) {
      continue;
    }

    way_handle* other = nullptr;
    while ((other = find_incident_line(*it, (**it).from_)) != nullptr) {
      if (*other == nullptr) {
        break;  // other alreay moved if join already discarded
      }
      if ((**it).oneway_ != (**other).oneway_) {
        break;  // dont join oneway with twoway
      }
      if ((**other).from_ == (**other).to_) {
        break;  // other is a "blossom"
      }

      auto joined = std::make_unique<way>();
      joined->to_ = (**it).to_;
      joined->oneway_ = (**it).oneway_;

      if ((**it).from_ == (**other).to_) {  //  --(other)--> X --(this)-->
        joined->from_ = (**other).from_;
      } else {  //  <--(other)-- X --(this)-->
        if ((**it).oneway_) {
          break;  // dont join conflicting oneway directions
        }
        joined->from_ = (**other).to_;
        mark_reversed(other->get());
      }

      joined->left_ = std::move(*other);
      joined->right_ = std::move(*it);
      *it = std::move(joined);
    }

    if ((**it).from_ == (**it).to_) {
      continue;  // cycle detected
    }

    while ((other = find_incident_line(*it, (**it).to_)) != nullptr) {
      if (*other == nullptr) {
        break;  // other alreay moved if join already discarded
      }
      if ((**it).oneway_ != (**other).oneway_) {
        break;  // dont join oneway with twoway
      }
      if ((**other).from_ == (**other).to_) {
        break;  // other is a "blossom"
      }

      auto joined = std::make_unique<way>();
      joined->from_ = (**it).from_;
      joined->oneway_ = (**it).oneway_;

      if ((**it).to_ == (**other).from_) {  // --(this)--> X --(other)-->
        joined->to_ = (**other).to_;
      } else {  // --(this)--> X <--(other)--
        if ((**it).oneway_) {
          break;  // conflicting oneway directions
        }
        joined->to_ = (**other).from_;
        mark_reversed(other->get());
      }

      joined->left_ = std::move(*it);
      joined->right_ = std::move(*other);
      *it = std::move(joined);
    }
  }
}

mcd::vector<osm_way> aggregate_geometry(std::vector<way_handle>& way_handles) {
  mcd::vector<osm_way> result;
  for (auto& wh : way_handles) {
    if (!wh) {  // joined away
      continue;
    } else if (wh->geometry_ != nullptr) {  // unjoined / single
      result.emplace_back(std::move(*wh->geometry_));
    } else {  // join result;
      osm_way joined_geo;
      joined_geo.oneway_ = wh->oneway_;

      std::stack<way_handle> stack;
      stack.emplace(std::move(wh));
      while (!stack.empty()) {
        auto curr = std::move(stack.top());
        stack.pop();

        if (curr->geometry_ != nullptr) {
          auto const skip = joined_geo.path_.size() == 0 ? 0 : 1;
          auto const& curr_geo = *curr->geometry_;

          if (curr->reversed_) {
            std::reverse_copy(begin(curr_geo.path_.polyline_),
                              std::next(end(curr_geo.path_.polyline_), -skip),
                              std::back_inserter(joined_geo.path_.polyline_));
            std::reverse_copy(
                begin(curr_geo.path_.osm_node_ids_),
                std::next(end(curr_geo.path_.osm_node_ids_), -skip),
                std::back_inserter(joined_geo.path_.osm_node_ids_));

          } else {
            std::copy(std::next(begin(curr_geo.path_.polyline_), skip),
                      end(curr_geo.path_.polyline_),
                      std::back_inserter(joined_geo.path_.polyline_));
            std::copy(std::next(begin(curr_geo.path_.osm_node_ids_), skip),
                      end(curr_geo.path_.osm_node_ids_),
                      std::back_inserter(joined_geo.path_.osm_node_ids_));
          }
          utl::concat(joined_geo.ids_, curr_geo.ids_);
        } else {
          if (curr->reversed_) {
            stack.emplace(std::move(curr->left_));
            stack.emplace(std::move(curr->right_));
          } else {
            stack.emplace(std::move(curr->right_));
            stack.emplace(std::move(curr->left_));
          }
        }
      }

      result.emplace_back(std::move(joined_geo));
    }
  }
  return result;
}

mcd::vector<osm_way> aggregate_osm_ways(mcd::vector<osm_way> ways) {
  auto way_handles = make_way_handles(ways);
  join_ways(way_handles);
  return aggregate_geometry(way_handles);
}

}  // namespace motis::path
