#pragma once

#include "motis/loader/loader_options.h"

namespace motis::test::schedule::rename_at_first_stop {

static loader::loader_options dataset_opt{"test/schedule/rename_at_first_stop",
                                          "20160128"};

constexpr auto kRisFolderArg =
    "--ris.input_folder=test/schedule/rename_at_first_stop/ris";

}  // namespace motis::test::schedule::rename_at_first_stop
