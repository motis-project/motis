#pragma once

#include <filesystem>

#include "motis/config.h"
#include "motis/data.h"

namespace motis {

data import(config const&,
            std::filesystem::path const& data_path,
            bool write = true);

}  // namespace motis