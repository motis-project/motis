#pragma once

#include <memory>
#include <string_view>

#include "ctx/res_id_t.h"

#include "motis/core/common/typed_flatbuffer.h"
#include "motis/core/common/unixtime.h"

#include "motis/core/schedule/free_text.h"
#include "motis/core/schedule/schedule.h"

#include "motis/module/message.h"

#include "motis/rt/delay_propagator.h"
#include "motis/rt/message_history.h"
#include "motis/rt/metrics.h"
#include "motis/rt/reroute.h"
#include "motis/rt/statistics.h"
#include "motis/rt/update_msg_builder.h"

namespace motis::rt {

struct rt_handler {
  explicit rt_handler(schedule& sched, ctx::res_id_t schedule_res_id,
                      bool validate_graph, bool validate_constant_graph,
                      bool print_stats, bool enable_history);

  motis::module::msg_ptr update(motis::module::msg_ptr const&);
  motis::module::msg_ptr single(motis::module::msg_ptr const&);
  motis::module::msg_ptr flush(motis::module::msg_ptr const&);

private:
  struct free_texts {
    trip const* trp_;
    free_text ft_;
    std::vector<ev_key> events_;
  };

  struct track_info {
    ev_key event_;
    std::string track_;
    motis::time schedule_time_;
  };

  void update(motis::ris::RISMessage const*, std::string_view msg_buffer,
              unixtime processing_time);
  void propagate();

  void batch_updates();

public:
  schedule& sched_;
  ctx::res_id_t schedule_res_id_;
  delay_propagator propagator_;
  update_msg_builder update_builder_;
  message_history msg_history_;
  statistics stats_;
  std::vector<track_info> track_events_;
  std::vector<free_texts> free_text_events_;
  std::map<schedule_event, delay_info*> cancelled_delays_;
  rt_metrics metrics_;

  bool validate_graph_;
  bool validate_constant_graph_;
  bool print_stats_;
};

}  // namespace motis::rt
