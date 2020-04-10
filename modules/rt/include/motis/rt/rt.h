#pragma once

#include <memory>
#include <vector>

#include "motis/module/module.h"

namespace motis::rt {

struct rt_handler;

struct rt : public motis::module::module {
  rt();
  ~rt() override;

  rt(rt const&) = delete;
  rt& operator=(rt const&) = delete;

  rt(rt&&) = delete;
  rt& operator=(rt&&) = delete;

  std::string name() const override { return "rt"; }

  void init(motis::module::registry&) override;

private:
  bool validate_graph_{true};
  bool validate_constant_graph_{false};

  std::unique_ptr<rt_handler> handler_;
};

}  // namespace motis::rt
