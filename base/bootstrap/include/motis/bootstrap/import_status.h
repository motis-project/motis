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
  int progress_{0U};
  std::string error_;
};

struct import_status : public module::progress_listener {
  bool update(motis::module::msg_ptr const&);
  void print();

  void update_progress(std::string const& task_name,
                       unsigned progress) override;
  void report_error(std::string const& task_name,
                    std::string const& what) override;

  bool silent_{false};

private:
  bool update(std::string const& task_name, state const& new_state);

  unsigned last_print_height_{0U};
  std::map<std::string, state> status_;
};

}  // namespace motis::bootstrap