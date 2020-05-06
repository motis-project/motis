#pragma once

#include "geo/polyline.h"

#include "motis/path/prepare/osm_path.h"
#include "motis/path/prepare/post/post_graph.h"
#include "motis/path/prepare/schedule/stations.h"

namespace motis::path {

struct seq_seg {
  seq_seg(uint32_t sequence, uint32_t segment)
      : sequence_(sequence), segment_(segment) {}

  friend bool operator==(seq_seg const& a, seq_seg const& b) {
    return std::tie(a.sequence_, a.segment_) ==
           std::tie(b.sequence_, b.segment_);
  }

  friend bool operator!=(seq_seg const& a, seq_seg const& b) {
    return std::tie(a.sequence_, a.segment_) !=
           std::tie(b.sequence_, b.segment_);
  }

  friend bool operator<(seq_seg const& a, seq_seg const& b) {
    return std::tie(a.sequence_, a.segment_) <
           std::tie(b.sequence_, b.segment_);
  }

  uint32_t sequence_, segment_;
};

struct db_builder {
  explicit db_builder(std::string const& fname);
  ~db_builder();

  db_builder(db_builder const&) noexcept = delete;  // NOLINT
  db_builder& operator=(db_builder const&) noexcept = delete;  // NOLINT
  db_builder(db_builder&&) noexcept = delete;  // NOLINT
  db_builder& operator=(db_builder&&) noexcept = delete;  // NOLINT

  void store_stations(std::vector<station> const&) const;

  std::pair<uint64_t, uint64_t> add_feature(
      geo::polyline const&, std::vector<seq_seg> const&,
      std::vector<uint32_t> const& classes, bool is_stub) const;

  void add_seq(size_t seq_idx, resolved_station_seq const&,
               std::vector<geo::box> const& boxes,
               std::vector<std::vector<int64_t>> const& feature_ids,
               std::vector<std::vector<uint64_t>> const& hints_rle) const;

  void finish() const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::path
