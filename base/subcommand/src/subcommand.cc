#include "motis/subcommand/subcommand.h"

#include "utl/verify.h"

#include <iostream>

namespace motis::bootstrap {

auto subcommands = std::map<std::string, subcommand_fn_t>{};

int run_subcommand(std::string const& name, int argc, char const** argv) {
  if (auto const it = subcommands.find(name); it != end(subcommands)) {
    return it->second(argc, argv);
  } else {
    std::cout << "no subcommand \"" << name << "\" available\n";
    return 1;
  }
}

subcommand_registration::subcommand_registration(std::string const& xname,
                                                 subcommand_fn_t fn) {
  std::cout << "registering subcommand " << xname << "\n";
  auto const inserted = subcommands.emplace(xname, std::move(fn)).second;
  utl::verify(inserted, "command {} did already exist", xname);
}

void list_subcommands() {
  std::cout << "Subcommands:\n";
  for (auto const& [name, cmd] : subcommands) {
    std::cout << "  " << name << "\n";
  }
}

}  // namespace motis::bootstrap