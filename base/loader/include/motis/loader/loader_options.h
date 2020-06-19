#pragma once

#include <ctime>
#include <string>
#include <utility>
#include <vector>

#include "motis/core/schedule/time.h"

namespace motis::loader {

struct loader_options {
  std::pair<std::time_t, std::time_t> interval() const;
  std::string graph_path() const;

  std::vector<std::string> dataset_{"rohdaten"};
  std::string schedule_begin_{"TODAY"};
  int num_days_{2};
  bool write_serialized_{false};
  bool write_graph_{false};
  bool read_graph_{false};
  bool read_graph_mmap_{false};
  bool apply_rules_{true};
  bool adjust_footpaths_{false};
  bool expand_trips_{true};
  bool expand_footpaths_{true};
  duration planned_transfer_delta_{30};
  std::string graph_path_{"default"};
  std::string wzr_classes_path_{""};
  std::string wzr_matrix_path_{""};

  std::vector<std::string> dataset_prefix_{};
};

}  // namespace motis::loader
