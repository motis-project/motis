#include "motis/module/subc_reg.h"

#include "utl/verify.h"

namespace motis::module {

void subc_reg::register_cmd(std::string const& name, std::string const& desc,
                            main_fn_t&& fn) {
  auto const inserted =
      commands_.emplace(name, subcommand{name, desc, std::move(fn)}).second;
  utl::verify(inserted, "subcommand name=\"{}\" already inserted", name);
}

void subc_reg::print_list() {
  if (commands_.empty()) {
    std::cout << "No commands available.\n";
    return;
  }

  auto const max_name_length =
      std::max_element(begin(commands_), end(commands_),
                       [](auto const& a, auto const& b) {
                         return a.second.name_.length() <
                                b.second.name_.length();
                       })
          ->second.name_.length();

  std::cout << "Available commands:\n";
  for (auto const& [name, c] : commands_) {
    std::cout << "  " << std::setw(max_name_length) << std::setfill(' ') << name
              << ": " << c.desc_ << "\n";
  }
  std::cout << "\n";
}

int subc_reg::execute(std::string const& name, int argc, char const** argv) {
  if (auto const it = commands_.find(name); it != end(commands_)) {
    return it->second.fn_(argc, argv);
  } else {
    std::cout << "command \"" << name << "\" not found\n";
    return 1;
  }
}

}  // namespace motis::module