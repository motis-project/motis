#pragma once

#include "motis/module/message.h"

namespace motis::paxforecast {
struct paxforecast;
}  // namespace motis::paxforecast

namespace motis::paxforecast::api {

motis::module::msg_ptr metrics(paxforecast& mod,
                               motis::module::msg_ptr const& msg);

}  // namespace motis::paxforecast::api
