#pragma once

#include "boost/asio/ssl/context.hpp"

namespace motis::launcher {

void load_server_certificate(boost::asio::ssl::context& ctx,
                             std::string const& cert_path,
                             std::string const& priv_key_path,
                             std::string const& dh_path);

}  // namespace motis::launcher