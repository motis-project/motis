#include <fstream>
#include <iostream>

#include "boost/url/url.hpp"

#include "utl/file_utils.h"
#include "utl/parser/cstr.h"

#include "./flags.h"

namespace po = boost::program_options;

namespace motis {

int params(int ac, char** av) {
  auto params = std::string{};
  auto in = std::string{};
  auto out = std::string{};

  auto desc = po::options_description{"Options"};
  desc.add_options()  //
      ("help", "Prints this help message")  //
      ("params,p", po::value(&params)->default_value(params))  //
      ("in,i", po::value(&in)->default_value(in))  //
      ("out,o", po::value(&out)->default_value(out));

  auto vm = parse_opt(ac, av, desc);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 0;
  }

  auto const override = boost::urls::url{params};

  auto out_file = std::ofstream{out};
  auto in_file = utl::open_file(in);
  auto line = std::optional<std::string>{};
  while ((line = utl::read_line(in_file))) {
    auto query = boost::urls::url{*line};
    for (auto const& x : override.params()) {
      query.params().set(x.key, x.value);
    }
    out_file << query << "\n";
  }

  return 0U;
}

}  // namespace motis