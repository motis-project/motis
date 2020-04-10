#pragma once

#include "ctx/ctx.h"

#include "motis/module/ctx_data.h"
#include "motis/module/dispatcher.h"

namespace motis {
namespace module {

inline boost::asio::io_service& get_io_service() {
  return ctx::current_op<ctx_data>().data_.dispatcher_->scheduler_.ios_;
}

}  // namespace module
}  // namespace motis
