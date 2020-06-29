#pragma once

#include "geo/polyline_format.h"

#include "motis/path/path_database_query.h"

namespace motis::path {

struct double_polyline_builder {
  void append(bool const is_fwd,
              path_database_query::resolvable_feature const* rf);

  [[nodiscard]] bool empty() const;
  void clear();
  void finish();

  bool is_extra_{false};
  std::vector<double> coords_;
};

struct google_polyline_builder {
  void append(bool const is_fwd,
              path_database_query::resolvable_feature const* rf);

  [[nodiscard]] bool empty() const;
  void clear();
  void finish();

  bool is_extra_{false};
  size_t count_{0};
  geo::polyline_encoder<6> enc_;
};

}  // namespace motis::path
