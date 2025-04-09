#pragma once

#include "pugixml.hpp"

#include "motis/vdv_rt/time.h"

namespace motis::vdv_rt {

pugi::xml_document make_xml_doc();

std::string xml_to_str(pugi::xml_document const& doc);

}  // namespace motis::vdv_rt