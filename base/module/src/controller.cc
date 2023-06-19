#include "motis/module/controller.h"

#include "motis/module/module.h"

namespace motis::module {

controller::controller(
    std::vector<std::unique_ptr<motis::module::module>>&& modules)
    : dispatcher{*static_cast<registry*>(this), std::move(modules)} {}

}  // namespace motis::module