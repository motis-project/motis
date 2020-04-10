#pragma once

#include <vector>

#include "motis/module/module.h"

namespace motis::ppr {

struct ppr : public motis::module::module {
  ppr();
  ~ppr() override;

  ppr(ppr const&) = delete;
  ppr& operator=(ppr const&) = delete;

  ppr(ppr&&) = delete;
  ppr& operator=(ppr&&) = delete;

  void init(motis::module::registry&) override;

private:
  std::string graph_file_{"routing-graph.ppr"};
  std::vector<std::string> profile_files_;
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  bool prefetch_rtrees_{true};
  bool verify_graph_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::ppr
