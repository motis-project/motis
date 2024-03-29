#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "motis/module/module.h"

namespace motis::routing {

struct memory;

struct routing : public motis::module::module {
  routing();
  ~routing() override;

  routing(routing const&) = delete;
  routing& operator=(routing const&) = delete;

  routing(routing&&) = delete;
  routing& operator=(routing&&) = delete;

  void reg_subc(motis::module::subc_reg&) override;
  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

private:
  motis::module::msg_ptr ontrip_train(motis::module::msg_ptr const&);
  motis::module::msg_ptr route(motis::module::msg_ptr const&);
  motis::module::msg_ptr trip_to_connection(motis::module::msg_ptr const&);

  std::mutex mem_pool_mutex_;
  std::vector<std::unique_ptr<memory>> mem_pool_;
  bool import_successful_{false};
};

}  // namespace motis::routing
