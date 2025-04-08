#include "motis/vdv_rt/client_status.h"

#include "fmt/format.h"

namespace motis::vdv_rt {
std::string client_status::operator()(boost::urls::url_view const&) const {
  fmt::println("client_status");
  return "client_status";
}
}  // namespace motis::vdv_rt