#pragma once

#include "boost/optional.hpp"

#include "utl/parser/cstr.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris::risml {

boost::optional<EventType> parse_type(
    utl::cstr const& raw,
    boost::optional<EventType> const& default_value = boost::none);

}  // namespace motis::ris::risml
