#pragma once

#include <string>

namespace motis::module {

struct progress_listener {
  virtual void set_progress_bounds(std::string const& name,  //
                                   float output_low, float output_high,
                                   float input_high) = 0;
  virtual void update_progress(std::string const& name, float progress) = 0;
  virtual void report_error(std::string const& name,
                            std::string const& what) = 0;
  virtual void report_step(std::string const& name,
                           std::string const& step) = 0;
};

}  // namespace motis::module