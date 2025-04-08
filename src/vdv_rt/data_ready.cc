#include "motis/vdv_rt/data_ready.h"

#include "fmt/format.h"

namespace motis::vdv_rt {
std::string data_ready::operator()(boost::urls::url_view const&) const {
  fmt::println("data_ready");
  return "data_ready";
}

}  // namespace motis::vdv_rt