#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>

#include "geo/latlng.h"

#include "utl/verify.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/schedule/station_sequences.h"
#include "motis/path/prepare/source_spec.h"

namespace motis::path {

using strategy_id_t = size_t;
constexpr auto kInvalidStrategyId = std::numeric_limits<strategy_id_t>::max();
using node_ref_id_t = size_t;

constexpr auto kInvalidNodeEquiv = std::numeric_limits<size_t>::max();

struct node_ref {
  node_ref() = default;
  node_ref(strategy_id_t const strategy_id, node_ref_id_t const id,
           geo::latlng const coords, double dist)
      : strategy_id_(strategy_id),
        id_(id),
        coords_(coords),
        dist_(dist),
        node_equiv_{kInvalidNodeEquiv} {}

  strategy_id_t strategy_id() const { return strategy_id_; };

  friend bool operator==(node_ref const& a, node_ref const& b) {
    return std::tie(a.id_, a.strategy_id_, a.coords_, a.dist_) ==
           std::tie(b.id_, b.strategy_id_, b.coords_, b.dist_);
  }

  friend bool operator<(node_ref const& a, node_ref const& b) {
    return std::tie(a.id_, a.strategy_id_, a.coords_, a.dist_) <
           std::tie(b.id_, b.strategy_id_, b.coords_, b.dist_);
  }

  strategy_id_t strategy_id_ = 0;
  node_ref_id_t id_ = 0;

  geo::latlng coords_;
  double dist_ = 0;
  size_t node_equiv_ = 0;  // equivalent to the ith node in a station nodes set
};

struct routing_result_matrix {
  routing_result_matrix(size_t from_size, size_t to_size)
      : from_size_{from_size},
        to_size_{to_size},
        data_(from_size_ * to_size_, std::numeric_limits<double>::infinity()) {}

  void verify_dimensions(size_t const from_size, size_t const to_size) const {
    utl::verify(from_size == from_size_ && to_size == to_size_,
                "routing_result_matrix: dimension verification failed");
  }

  template <typename Fn>
  void foreach (Fn&& fn) const {
    for (auto from_idx = 0UL; from_idx < from_size_; ++from_idx) {
      for (auto to_idx = 0UL; to_idx < to_size_; ++to_idx) {
        fn(from_idx, to_idx, this->operator()(from_idx, to_idx));
      }
    }
  }

  template <typename Fn>
  void foreach (Fn&& fn) {
    for (auto from_idx = 0UL; from_idx < from_size_; ++from_idx) {
      for (auto to_idx = 0UL; to_idx < to_size_; ++to_idx) {
        fn(from_idx, to_idx, this->operator()(from_idx, to_idx));
      }
    }
  }

  double const& operator()(size_t f, size_t t) const {
    return data_[f * to_size_ + t];
  }
  double& operator()(size_t f, size_t t) { return data_[f * to_size_ + t]; }

  void transpose() {
    for (auto f = 0UL; f < from_size_; ++f) {
      for (auto t = 0UL; t < f; ++t) {
        std::swap(data_[t * from_size_ + f], data_[f * to_size_ + t]);
      }
    }
    std::swap(from_size_, to_size_);
  }

  size_t from_size_, to_size_;
  std::vector<double> data_;
};

struct routing_strategy {
  routing_strategy(strategy_id_t strategy_id, source_spec spec)
      : strategy_id_{strategy_id}, source_spec_{spec} {}
  virtual ~routing_strategy() = default;

  routing_strategy(routing_strategy const&) noexcept = delete;
  routing_strategy& operator=(routing_strategy const&) noexcept = delete;
  routing_strategy(routing_strategy&&) noexcept = delete;
  routing_strategy& operator=(routing_strategy&&) noexcept = delete;

  virtual std::vector<node_ref> const& close_nodes(
      std::string const& station_id) const = 0;

  virtual routing_result_matrix find_routes(
      std::vector<node_ref> const& from,
      std::vector<node_ref> const& to) const = 0;

  virtual osm_path get_path(node_ref const& from, node_ref const& to) const = 0;

  strategy_id_t strategy_id_;
  source_spec source_spec_;
};

}  // namespace motis::path
