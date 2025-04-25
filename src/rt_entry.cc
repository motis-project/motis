#include "motis/rt_entry.h"

namespace motis {

std::variant<rt_entry::gtfsrt, rt_entry::vdvaus> rt_entry::operator()() const {
  if (!protocol_ || protocol_ == rt_entry::protocol::gtfsrt) {
    return rt_entry::gtfsrt{url_, headers_ ? &(*headers_) : nullptr};
  } else {
    return rt_entry::vdvaus{url_, *server_name_, *client_name_, *hysteresis_};
  }
}

}  // namespace motis