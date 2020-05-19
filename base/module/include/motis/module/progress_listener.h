#pragma once

#include <string>

namespace motis::module {

struct progress_listener {
  // virtual ~progress_listener() = 0;

  virtual void set_progress_bounds(std::string const& name,  //
                                   double output_low, double output_high,
                                   double input_high) = 0;
  virtual void update_progress(std::string const& name, double progress) = 0;
  virtual void report_error(std::string const& name,
                            std::string const& what) = 0;
  virtual void report_step(std::string const& name,
                           std::string const& task) = 0;
};

}  // namespace motis::module