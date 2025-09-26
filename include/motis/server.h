#pragma once

#include <string_view>

#include "boost/url/url_view.hpp"

namespace motis {

struct data;
struct config;

int server(data d, config const& c, std::string_view);

unsigned get_api_version(boost::urls::url_view const&);

}  // namespace motis