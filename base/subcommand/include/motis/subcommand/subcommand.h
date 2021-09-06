#pragma once

#include <functional>
#include <map>
#include <string>

namespace motis::bootstrap {

using subcommand_fn_t =
    std::function<int(int /* argc */, char const** /*argv */)>;

struct subcommand_registration {
  subcommand_registration(std::string const& name, subcommand_fn_t);
};

int run_subcommand(std::string const&, int argc, char const** argv);

void list_subcommands();

}  // namespace motis::bootstrap