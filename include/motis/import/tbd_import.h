#pragma once

#include "motis/config.h"
#include "motis/import/dataset_hashes.h"
#include "motis/import/task.h"

namespace motis {

struct tbd_import : public task {
  tbd_import(std::filesystem::path const& data_path,
             config const&,
             dataset_hashes const&);
  ~tbd_import() override;
  void run() override;
  bool is_enabled() const override;
};

}  // namespace motis
