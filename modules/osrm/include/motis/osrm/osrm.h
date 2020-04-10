#pragma once

#include <map>
#include <memory>
#include <vector>

#include "motis/module/module.h"

namespace motis::osrm {
struct router;

struct osrm : public motis::module::module {
public:
  osrm();
  ~osrm() override;

  osrm(osrm const&) = delete;
  osrm& operator=(osrm const&) = delete;

  osrm(osrm&&) = delete;
  osrm& operator=(osrm&&) = delete;

  std::string name() const override { return "osrm"; }
  void init(motis::module::registry&) override;
  void init_async();

private:
  router const* get_router(std::string const& profile);

  std::vector<std::string> datasets_;
  std::map<std::string, std::unique_ptr<router>> routers_;
};

}  // namespace motis::osrm
