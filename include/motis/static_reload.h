#pragma once

#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

#include "motis/config.h"

namespace motis {

std::vector<std::string> due_static_reloads(
    config::timetable const&,
    std::map<std::string, std::chrono::system_clock::time_point> const&
        last_fired,
    std::chrono::system_clock::time_point now);

boost::asio::awaitable<void> reload_static_datasets(
    config const&,
    std::filesystem::path data_path,
    std::vector<std::string> dataset_tags);

void run_static_reload(boost::asio::io_context&,
                       config const&,
                       std::filesystem::path data_path,
                       std::function<void()> on_reloaded);

}  // namespace motis
