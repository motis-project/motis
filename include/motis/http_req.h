#pragma once

#include <chrono>
#include <map>
#include <string>

#include "boost/asio/awaitable.hpp"
#include "boost/beast/http/dynamic_body.hpp"
#include "boost/beast/http/message.hpp"
#include "boost/url/url.hpp"

namespace motis {

using http_response =
    boost::beast::http::response<boost::beast::http::dynamic_body>;

boost::asio::awaitable<http_response> http_GET(
    boost::urls::url,
    std::map<std::string, std::string> const& headers,
    std::chrono::seconds timeout);

}  // namespace motis