#pragma once

#include <map>
#include <string>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/co_spawn.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/beast/core.hpp"
#include "boost/beast/http.hpp"
#include "boost/beast/version.hpp"
#include "boost/url/url.hpp"

namespace motis {

using http_response =
    boost::beast::http::response<boost::beast::http::dynamic_body>;

boost::asio::awaitable<http_response> http_GET(
    boost::urls::url, std::map<std::string, std::string> const& headers);

}  // namespace motis