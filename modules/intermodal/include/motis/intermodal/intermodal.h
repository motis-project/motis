#pragma once

#include <string>

#include "motis/module/module.h"
#include "motis/intermodal/ppr_profiles.h"

namespace motis::intermodal {

struct intermodal : public motis::module::module {
public:
  intermodal();
  ~intermodal() override;

  intermodal(intermodal const&) = delete;
  intermodal& operator=(intermodal const&) = delete;

  intermodal(intermodal&&) = delete;
  intermodal& operator=(intermodal&&) = delete;

  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr route(motis::module::msg_ptr const&);

  std::string router_{"routing"};
  bool revise_{false};
  ppr_profiles ppr_profiles_;
};

}  // namespace motis::intermodal
