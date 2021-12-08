#include "motis/module/module.h"

#include "boost/filesystem.hpp"

#include "motis/core/schedule/schedule.h"

namespace fs = boost::filesystem;

namespace motis::module {

schedule const& module::get_sched() const {
  return *get_shared_data<schedule_data>(to_res_id(global_res_id::SCHEDULE))
              .schedule_;
}

void module::set_data_directory(std::string const& d) { data_directory_ = d; }

void module::set_shared_data(ctx::access_scheduler<ctx_data>* d) {
  shared_data_ = d;
}

std::string module::data_path(fs::path const& p) {
  return p.parent_path() == data_directory_
             ? p.lexically_relative(data_directory_).generic_string()
             : p.generic_string();
}

fs::path const& module::get_data_directory() const { return data_directory_; }

}  // namespace motis::module
