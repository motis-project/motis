#pragma once

#include <optional>

#include "boost/beast/http/string_body.hpp"

namespace motis {

using http_req_t = boost::beast::http::request<boost::beast::http::string_body>;
using http_res_t =
    boost::beast::http::response<boost::beast::http::string_body>;
using http_res_cb_t = std::function<void(http_res_t&&)>;
using http_handler_t = std::function<void(http_req_t, http_res_cb_t&&)>;

void serve(std::string const& host,
           std::uint_least16_t const port,
           std::string const& path,
           http_handler_t const&);

}  // namespace motis