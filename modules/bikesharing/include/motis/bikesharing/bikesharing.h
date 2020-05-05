#pragma once

#include <string>
#include <vector>

#include "motis/module/module.h"

namespace motis::bikesharing {

struct database;
struct geo_index;

struct bikesharing : public motis::module::module {
  bikesharing();
  ~bikesharing() override;

  bikesharing(bikesharing const&) = delete;
  bikesharing& operator=(bikesharing const&) = delete;

  bikesharing(bikesharing&&) = delete;
  bikesharing& operator=(bikesharing&&) = delete;

  void init(motis::module::registry&) override;

private:
  void init_module();
  motis::module::msg_ptr search(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr geo_terminals(motis::module::msg_ptr const&) const;

  void ensure_initialized() const;

  std::string database_path_{"bikesharing.mdb"};
  std::string nextbike_path_;
  size_t db_max_size_{static_cast<size_t>(1024) * 1024 * 1024 * 512};

  std::unique_ptr<database> database_;
  std::unique_ptr<geo_index> geo_index_;
};

}  // namespace motis::bikesharing
