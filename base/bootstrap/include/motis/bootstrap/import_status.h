#pragma once

#include <map>
#include <string>
#include <vector>

#include "cista/reflection/comparable.h"

#include "motis/module/message.h"

namespace motis::bootstrap {

struct state {
  CISTA_COMPARABLE()
  std::vector<std::string> dependencies_;
  import::Status status_{import::Status::Status_WAITING};
  int progress_{0U};
};

struct import_status {
  bool update(motis::module::msg_ptr const&);
  void print();

  unsigned last_print_height_{0U};
  std::map<std::string, state> status_;
};

}  // namespace motis::bootstrap