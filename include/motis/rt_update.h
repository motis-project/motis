#pragma once

#include "boost/asio/awaitable.hpp"

#include "motis/fwd.h"

namespace motis {

boost::asio::awaitable<void> rt_update(config const&,
                                       nigiri::timetable const&,
                                       tag_lookup const& tags,
                                       std::shared_ptr<rt>&);

}