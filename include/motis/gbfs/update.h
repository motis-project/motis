#pragma once

#include <memory>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"

namespace motis::gbfs {

boost::asio::awaitable<void> update(config const&,
                                    osr::ways const&,
                                    osr::lookup const&,
                                    std::shared_ptr<gbfs_data>&);

void run_gbfs_update(boost::asio::io_context&,
                     config const&,
                     osr::ways const&,
                     osr::lookup const&,
                     std::shared_ptr<gbfs_data>&);

}  // namespace motis::gbfs
