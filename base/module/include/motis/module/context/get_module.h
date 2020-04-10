#pragma once

#include <string_view>

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"

namespace motis::module {

template <typename Module>
Module& get_module(std::string_view module_name) {
  return current_data().dispatcher_->get_module<Module>(module_name);
}

}  // namespace motis::module
