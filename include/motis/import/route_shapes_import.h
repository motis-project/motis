#pragma once

#include <array>

#include "nigiri/clasz.h"

#include "motis/config.h"
#include "motis/import/dataset_hashes.h"
#include "motis/import/task.h"

namespace motis {

struct route_shapes_import : public task {
  static std::array<bool, nigiri::kNumClasses> get_clasz_enabled(config const&);
  static bool get_reuse_shapes_cache(std::filesystem::path const& data_path,
                                     config const&,
                                     dataset_hashes const&);
  static bool get_keep_routed_shape_data(std::filesystem::path const& data_path,
                                         config const&,
                                         dataset_hashes const&);
  static void cleanup_stale_cache(std::filesystem::path const& data_path);

  route_shapes_import(
      std::filesystem::path const& data_path,
      config const&,
      dataset_hashes const&);
  ~route_shapes_import() override;
  void run() override;
  bool is_enabled() const override;

  std::array<bool, nigiri::kNumClasses> route_shapes_clasz_enabled_;
  bool reuse_shapes_cache_{false};
};

}  // namespace motis
