#include "motis/module/module.h"

#include "motis/module/dispatcher.h"

namespace fs = std::filesystem;

namespace motis::module {

void module::set_data_directory(std::string const& d) { data_directory_ = d; }

void module::set_shared_data(dispatcher* d) { shared_data_ = d; }

locked_resources module::lock_resources(ctx::accesses_t access,
                                        ctx::op_type_t op_type) {
  if (dispatcher::direct_mode_dispatcher_ != nullptr) {
    return {{shared_data_}};
  } else {
    return {{ctx::access_scheduler<ctx_data>::mutex{*shared_data_, op_type,
                                                    std::move(access)}}};
  }
}

std::string module::data_path(fs::path const& p) const {
  return p.parent_path() == data_directory_
             ? p.lexically_relative(data_directory_).generic_string()
             : p.generic_string();
}

fs::path const& module::get_data_directory() const { return data_directory_; }

}  // namespace motis::module
