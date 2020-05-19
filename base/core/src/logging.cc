#include "motis/core/common/logging.h"

namespace motis::logging {

std::mutex log::log_mutex_;
bool log::enabled_ = true;

void clog_import_step(std::string const& name, double output_low,
                      double output_high, double input_high) {
  std::clog << '\0' << 'S' << name << '\0'  //
            << '\0' << 'B' << output_low << " " << output_high << " "
            << input_high << '\0';
}

void clog_import_progress(size_t progress, size_t rate_limit) {
  if (progress % rate_limit == 0) {
    std::clog << '\0' << progress << '\0';
  }
}

}  // namespace motis::logging