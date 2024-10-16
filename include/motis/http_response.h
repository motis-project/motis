#pragma once

#include "boost/beast/http/string_body.hpp"

namespace motis {

using http_response =
    boost::beast::http::response<boost::beast::http::string_body>;

}  // namespace motis