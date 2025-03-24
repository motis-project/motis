#pragma once

#include <string_view>

namespace motis {

struct data;
struct config;

int server(data d, config const& c, std::string_view);

}  // namespace motis