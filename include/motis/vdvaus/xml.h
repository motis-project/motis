#pragma once

#include "pugixml.hpp"

#include "motis/vdvaus/time.h"

namespace motis::vdvaus {

pugi::xml_document make_xml_doc();

std::string xml_to_str(pugi::xml_document const& doc);

pugi::xml_document parse(std::string const&);

}  // namespace motis::vdvaus