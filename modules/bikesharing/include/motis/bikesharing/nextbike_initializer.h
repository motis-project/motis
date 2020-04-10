#pragma once

#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "motis/module/message.h"

#include "motis/bikesharing/database.h"
#include "motis/bikesharing/terminal.h"

#include "utl/parser/buffer.h"

#include "motis/bikesharing/terminal.h"

namespace motis::bikesharing {

using close_locations = std::vector<std::vector<close_location>>;

void initialize_nextbike(std::string const& nextbike_path, database&);

std::pair<std::vector<terminal>, std::vector<hourly_availabilities>>
load_and_merge(std::string const& nextbike_path);

std::vector<std::string> get_nextbike_files(std::string const&);
std::time_t nextbike_filename_to_timestamp(std::string const&);
std::vector<terminal_snapshot> nextbike_parse_xml(utl::buffer&&);

close_locations find_close_stations(std::vector<terminal> const&);
motis::module::msg_ptr to_geo_request(std::vector<terminal> const&, double r);

close_locations find_close_terminals(std::vector<terminal> const& terminals);

}  // namespace motis::bikesharing
