#pragma once

#include <memory>

#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::gbfs {

void run_gbfs_update(boost::asio::io_context&,
                     config const&,
                     osr::ways const&,
                     osr::lookup const&,
                     std::shared_ptr<gbfs_data>&);

}  // namespace motis::gbfs
