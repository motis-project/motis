#include "motis/module/module.h"

#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

namespace motis::module {

void module::set_data_directory(std::string const& d) { data_directory_ = d; }

void module::set_context(motis::schedule& schedule) { schedule_ = &schedule; }

std::string module::data_path(fs::path const& p) {
  return p.parent_path() == data_directory_
             ? p.lexically_relative(data_directory_).generic_string()
             : p.generic_string();
}

fs::path const& module::get_data_directory() const { return data_directory_; }

}  // namespace motis::module