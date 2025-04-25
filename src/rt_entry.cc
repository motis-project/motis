#include "motis/rt_ep_config.h"

namespace motis {

std::variant<rt_ep_config::gtfsrt, rt_ep_config::vdvaus>
rt_ep_config::operator()() const {
  if (!protocol_ || protocol_ == rt_ep_config::protocol::gtfsrt) {
    return rt_ep_config::gtfsrt{url_, headers_ ? &(*headers_) : nullptr};
  } else {
    return rt_ep_config::vdvaus{url_, *server_name_, *client_name_,
                                *hysteresis_};
  }
}

}  // namespace motis