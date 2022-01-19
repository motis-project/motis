#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "ctx/res_id_t.h"

#include "motis/module/module.h"

namespace motis {
struct schedule;
}

namespace motis::rt {

struct rt_handler;

struct rt : public motis::module::module {
  rt();
  ~rt() override;

  rt(rt const&) = delete;
  rt& operator=(rt const&) = delete;

  rt(rt&&) = delete;
  rt& operator=(rt&&) = delete;

  void init(motis::module::registry&) override;

private:
  rt_handler& get_or_create_rt_handler(schedule& sched,
                                       ctx::res_id_t schedule_res_id);

  bool validate_graph_{false};
  bool validate_constant_graph_{false};
  bool print_stats_{true};

  std::mutex handler_mutex;
  std::map<ctx::res_id_t, std::unique_ptr<rt_handler>> handlers_;
};

}  // namespace motis::rt
