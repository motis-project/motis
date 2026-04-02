#include "motis/import/task.h"

#include "utl/helpers/algorithm.h"

namespace motis {

task::task(std::filesystem::path const& data_path, data& d, config& c)
    : data_path_{data_path}, c_{c}, d_{d} {}

task::~task() = default;

std::string_view task::name() const { return name_; }

bool task::is_done() const { return done_; }

bool task::is_ready_to_run() const {
  return utl::all_of(in_, [](task const* t) { return t->is_done(); });
}

void task::exec() {
  if (!is_enabled()) {
    return;
  }

  if (can_load()) {
    load();
  } else {
    run();
  }

  done_ = true;
}

}  // namespace motis