#include "motis/rt_entry.h"

namespace motis {

std::variant<rt_entry::gtfs_rt, rt_entry::vdv_rt> rt_entry::operator()() const {
  if (!protocol_ || protocol_ == rt_entry::protocol::gtfs_rt) {
    return rt_entry::gtfs_rt{url_, headers_ ? &(*headers_) : nullptr};
  } else {
    return rt_entry::vdv_rt{url_, *server_name_, *client_name_, *hysteresis_};
  }
}

}  // namespace motis