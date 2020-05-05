#include "motis/module/module.h"

#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

namespace motis::module {

void module::set_data_directory(std::string const& d) { data_directory_ = d; }

void module::set_context(motis::schedule& schedule) { schedule_ = &schedule; }

fs::path const& module::get_data_directory() const { return data_directory_; }

}  // namespace motis::module