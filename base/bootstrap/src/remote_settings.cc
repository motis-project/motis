#include "motis/bootstrap/remote_settings.h"

#include "utl/verify.h"

namespace motis::bootstrap {

std::vector<std::pair<std::string, std::string>> remote_settings::get_remotes()
    const {
  std::vector<std::pair<std::string, std::string>> remotes;
  for (auto const& r : remotes_) {
    auto const split_pos = r.find(':');
    utl::verify(split_pos != std::string::npos, "invalid remote");

    auto const host = r.substr(0, split_pos);
    auto const port = r.substr(split_pos + 1);
    utl::verify(!host.empty() && !port.empty() &&
                    std::all_of(begin(port), end(port),
                                [](auto c) { return std::isdigit(c); }),
                "invalid remote");
    remotes.emplace_back(host, port);
  }
  return remotes;
}

}  // namespace motis::bootstrap
