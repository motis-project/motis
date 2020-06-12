#pragma once

#include "motis/bootstrap/motis_instance.h"
#include "motis/loader/loader_options.h"

namespace motis::bootstrap {

void register_import_schedule(motis_instance&, loader::loader_options const&,
                              std::string data_dir);

}  // namespace motis::bootstrap
