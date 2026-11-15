#pragma once

#include <chrono>
#include <memory>

#include "motis/fwd.h"

namespace boost::asio {
class io_context;
}

namespace motis {

void run_rt_update(boost::asio::io_context&, config const&, data&);

void apply_canned_rt_update(config const&, data&);

bool finalize_rtt(nigiri::timetable const&, nigiri::rt_timetable&);

}  // namespace motis