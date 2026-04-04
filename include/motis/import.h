#pragma once

#include <filesystem>

#include "motis/config.h"

namespace motis {

void import(config const&, std::filesystem::path const& data_path);

}  // namespace motis
