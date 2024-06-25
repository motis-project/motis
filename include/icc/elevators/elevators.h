#pragma once

#include <memory>
#include <string_view>

#include "osr/ways.h"

#include "icc/elevators/match_elevator.h"
#include "icc/elevators/parse_fasta.h"

namespace icc {

struct shared_elevators {
  struct elevators {
    elevators(osr::ways const& w,
              hash_set<osr::node_idx_t> const& elevator_nodes,
              vector_map<elevator_idx_t, elevator>&& elevators)
        : elevators_{std::move(elevators)},
          elevators_rtree_{create_elevator_rtree(elevators_)},
          blocked_{get_blocked_elevators(
              w, elevators_, elevators_rtree_, elevator_nodes)} {}

    vector_map<elevator_idx_t, elevator> elevators_;
    point_rtree<elevator_idx_t> elevators_rtree_;
    osr::bitvec<osr::node_idx_t> blocked_;
  };

  shared_elevators(osr::ways const& w,
                   hash_set<osr::node_idx_t> const& elevator_nodes,
                   vector_map<elevator_idx_t, elevator>&& e)
      : e_{std::make_shared<elevators>(w, elevator_nodes, std::move(e))} {}

  void set(elevators&& upd) {
    auto l = std::lock_guard{m_};
    e_ = std::make_shared<elevators>(std::move(upd));
  }

  std::shared_ptr<elevators> get() const {
    std::shared_ptr<elevators> copy;
    {
      auto const lock = std::lock_guard{m_};
      copy = e_;
    }
    return copy;
  }

private:
  std::shared_ptr<elevators> e_;
  mutable std::mutex m_;
};

}  // namespace icc