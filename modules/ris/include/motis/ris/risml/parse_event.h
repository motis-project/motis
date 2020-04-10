#pragma once

#include "boost/optional.hpp"

#include "pugixml.hpp"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::ris::risml {

struct context;

flatbuffers::Offset<AdditionalEvent> parse_additional_event(
    flatbuffers::FlatBufferBuilder&, flatbuffers::Offset<Event> const&,
    pugi::xml_node const& e_node, pugi::xml_node const& t_node);

boost::optional<flatbuffers::Offset<Event>> parse_standalone_event(
    context&, pugi::xml_node const& e_node);

}  // namespace motis::ris::risml
