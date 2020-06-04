#pragma once

#include <map>
#include <string>
#include <vector>

#include "motis/module/module.h"

#include "motis/ppr/profile_info.h"

namespace motis::ppr {

struct ppr : public motis::module::module {
  ppr();
  ~ppr() override;

  ppr(ppr const&) = delete;
  ppr& operator=(ppr const&) = delete;

  ppr(ppr&&) = delete;
  ppr& operator=(ppr&&) = delete;

  void import(motis::module::registry& reg) override;
  void init(motis::module::registry&) override;

  bool import_successful() const override { return import_successful_; }

  std::map<std::string, profile_info> profiles_;

private:
  std::string graph_file() const;

  std::vector<std::string> profile_files_;
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  bool prefetch_rtrees_{true};
  bool verify_graph_{false};

  bool use_dem_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
  bool import_successful_{false};
};

}  // namespace motis::ppr
