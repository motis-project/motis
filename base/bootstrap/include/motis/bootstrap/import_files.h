#pragma once

#include "motis/module/registry.h"

namespace motis::bootstrap {

motis::module::msg_ptr make_file_event(
    std::vector<std::string> const& import_paths);

void register_import_files(motis::module::registry&);

}  // namespace motis::bootstrap
