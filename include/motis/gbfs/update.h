#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "boost/asio/awaitable.hpp"
#include "boost/asio/io_context.hpp"

#include "motis/fwd.h"
#include "motis/metrics_registry.h"

namespace motis::gbfs {

std::string normalize_gbfs_url(std::string_view);

boost::asio::awaitable<void> update(config const&,
                                    osr::ways const&,
                                    osr::lookup const&,
                                    std::shared_ptr<gbfs_data>&,
                                    metrics_registry const*,
                                    feed_cache* = nullptr);

void apply_canned_gbfs_update(config const&, data&);

void run_gbfs_update(boost::asio::io_context&,
                     config const&,
                     osr::ways const&,
                     osr::lookup const&,
                     std::shared_ptr<gbfs_data>&,
                     metrics_registry const*);

}  // namespace motis::gbfs
