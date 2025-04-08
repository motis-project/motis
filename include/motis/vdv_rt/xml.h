#pragma once

#include "pugixml.hpp"

#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

pugi::xml_document make_xml_doc();

void add_status_node(pugi::xml_node& node);

void add_start_time_node(pugi::xml_node&, sys_time start);

void add_ack_node(pugi::xml_node& node);

std::string xml_to_str(pugi::xml_document const& doc);

}  // namespace motis::vdv_rt