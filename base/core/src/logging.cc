#include "motis/core/common/logging.h"

namespace motis::logging {

std::mutex log::log_mutex_;
bool log::enabled_ = false;

}  // namespace motis::logging