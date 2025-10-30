#pragma once

#include <chrono>
#include <map>
#include <string>

#include "boost/asio/awaitable.hpp"
#include "boost/beast/http/dynamic_body.hpp"
#include "boost/beast/http/message.hpp"
#include "boost/url/url.hpp"

namespace motis {

constexpr auto const kBodySizeLimit = 512U * 1024U * 1024U;  // 512 M

using http_response =
    boost::beast::http::response<boost::beast::http::dynamic_body>;

boost::asio::awaitable<http_response> http_GET(
    boost::urls::url,
    std::map<std::string, std::string> const& headers,
    std::chrono::seconds timeout);

boost::asio::awaitable<http_response> http_POST(
    boost::urls::url,
    std::map<std::string, std::string> const& headers,
    std::string const& body,
    std::chrono::seconds timeout);

std::string get_http_body(http_response const&);

}  // namespace motis
