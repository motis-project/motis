#pragma once

#include "motis/module/message.h"
#include "motis/bootstrap/import_settings.h"
#include "motis/bootstrap/module_settings.h"
#include "motis/bootstrap/motis_instance.h"
#include "motis/loader/loader_options.h"

namespace motis::bootstrap {

motis::module::msg_ptr import_schedule(import_settings const&,
                                       module_settings const&,
                                       loader::loader_options const&,
                                       motis::module::msg_ptr const&,
                                       motis_instance&);

}  // namespace motis::bootstrap
