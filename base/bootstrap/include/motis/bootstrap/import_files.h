#pragma once

#include "motis/module/import_dispatcher.h"

namespace motis::bootstrap {

std::tuple<std::string, std::string, std::string> split_import_path(
    std::string const& import_path);

motis::module::msg_ptr make_file_event(
    std::vector<std::string> const& import_paths);

void register_import_files(motis::module::import_dispatcher&);

}  // namespace motis::bootstrap
