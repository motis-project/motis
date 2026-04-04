#pragma once

#include "motis/config.h"
#include "motis/import/dataset_hashes.h"
#include "motis/import/task.h"

namespace motis {

struct tt_import : public task {
  tt_import(std::filesystem::path const& data_path,
            config const&,
            dataset_hashes const&);
  ~tt_import() override;
  void run() override;
  bool is_enabled() const override;

  bool keep_routed_shape_data_;
};

}  // namespace motis
