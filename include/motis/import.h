#pragma once

#include <filesystem>

#include "motis/config.h"

namespace motis {

void import(config const&,
            std::filesystem::path const& data_path,
            std::optional<std::vector<std::string>> const& task_filter = {});

}  // namespace motis
