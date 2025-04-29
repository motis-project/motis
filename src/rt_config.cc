#include "motis/rt_config.h"

namespace motis {

std::variant<rt_config::gtfsrt, rt_config::vdvaus> rt_config::operator()()
    const {
  switch (protocol_) {
    case protocol::gtfsrt:
      return rt_config::gtfsrt{url_, headers_ ? &(*headers_) : nullptr};
    case protocol::vdvaus:
      return rt_config::vdvaus{url_, *server_name_, *client_name_,
                               *hysteresis_};
  }
  std::unreachable();
}

}  // namespace motis