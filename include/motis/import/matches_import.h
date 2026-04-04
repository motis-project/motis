#pragma once

#include "motis/config.h"
#include "motis/import/dataset_hashes.h"
#include "motis/import/task.h"

namespace motis {

struct matches_import : public task {
  matches_import(std::filesystem::path const& data_path,
                 config const&,
                 dataset_hashes const&);
  ~matches_import() override;
  void run() override;
  bool is_enabled() const override;
};

}  // namespace motis
