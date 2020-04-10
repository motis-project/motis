#pragma once

#include <string>

#include "conf/configuration.h"

namespace motis::launcher {

struct server_settings : public conf::configuration {
  server_settings() : configuration("Listener Options", "server") {
    param(host_, "host", "host (e.g. 0.0.0.0 or localhost)");
    param(port_, "port", "port (e.g. https or 8443)");

#if defined(NET_TLS)
    param(cert_path_, "cert_path",
          "certificate path or ::dev:: for hardcoded cert");
    param(priv_key_path_, "priv_key_path",
          "private key path or ::dev:: for hardcoded cert");
    param(dh_path_, "dh_path", "dh parameters path or ::dev:: for hardcoded");
#endif

    param(api_key_, "api_key", "API key (empty = no protection)");
    param(log_path_, "log_path", "log requests to file (empty = no logging)");
    param(static_path_, "static_path", "path to ui/web (compiled)");
  }

  std::string host_{"0.0.0.0"}, port_{"8080"};
#if defined(NET_TLS)
  std::string cert_path_{"::dev::"}, priv_key_path_{"::dev::"},
      dh_path_{"::dev::"};
#endif
  std::string api_key_;
  std::string log_path_;
  std::string static_path_;
};

}  // namespace motis::launcher
