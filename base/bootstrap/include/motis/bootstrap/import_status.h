#pragma once

#include <map>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/module/message.h"
#include "motis/module/progress_listener.h"

namespace motis::bootstrap {

struct state {
  CISTA_COMPARABLE()
  std::vector<std::string> dependencies_;
  import::Status status_{import::Status::Status_WAITING};
  float progress_{0.F}, output_low_{0.F}, output_high_{100.F},
      input_high_{100.F};
  std::string error_;
  std::string current_task_;
};

struct import_status : public module::progress_listener {
  bool update(motis::module::msg_ptr const&);
  void print();

  void set_progress_bounds(std::string const& name, float output_low,
                           float output_high, float input_high) override;
  void update_progress(std::string const& name, float progress) override;

  void report_error(std::string const& name, std::string const& what) override;
  void report_step(std::string const& name, std::string const& step) override;

  bool silent_{false};

private:
  bool update(std::string const& task_name, state const& new_state);

  unsigned last_print_height_{0U};
  std::map<std::string, state> status_;
};

}  // namespace motis::bootstrap