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
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override;

private:
  rt_handler& get_or_create_rt_handler(schedule& sched,
                                       ctx::res_id_t schedule_res_id);
  rt_handler* get_rt_handler(ctx::res_id_t schedule_res_id);

  bool validate_graph_{false};
  bool validate_constant_graph_{false};
  bool print_stats_{true};
  bool enable_history_{false};

  std::mutex handler_mutex_;
  std::map<ctx::res_id_t, std::unique_ptr<rt_handler>> handlers_;

  bool import_successful_{false};
};

}  // namespace motis::rt
