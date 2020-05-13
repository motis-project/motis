#pragma once

#include <string>

namespace motis::module {

struct progress_listener {
  virtual void update_progress(std::string const& task_name,
                               unsigned progress) = 0;
  virtual void report_error(std::string const& task_name,
                            std::string const& what) = 0;
};

}  // namespace motis::module