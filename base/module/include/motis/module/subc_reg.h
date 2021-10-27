#pragma once

#include <functional>
#include <map>
#include <string>

namespace motis::module {

using main_fn_t = std::function<int(int, char const**)>;

struct subcommand {
  std::string name_, desc_;
  main_fn_t fn_;
};

struct subc_reg {
  void register_cmd(std::string const& name, std::string const& desc,
                    main_fn_t&&);
  void print_list();
  int execute(std::string const& name, int argc, char const** argv);
  std::map<std::string, subcommand> commands_;
};

}  // namespace motis::module