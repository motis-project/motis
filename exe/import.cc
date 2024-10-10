#include <filesystem>

#include "utl/progress_tracker.h"

#include "boost/program_options.hpp"

#include "motis/config.h"
#include "motis/import.h"

namespace bpo = boost::program_options;
namespace fs = std::filesystem;

namespace motis {

int import(int ac, char** av) {
  auto config_path = fs::path{"config.yml"};
  auto data_path = fs::path{"data"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("config,c", bpo::value(&config_path)->default_value(config_path),
       "configuration file")  //
      ("data,d", bpo::value(&data_path)->default_value(data_path), "data path");

  auto vm = bpo::variables_map{};
  bpo::store(bpo::command_line_parser(ac, av).options(desc).run(), vm);
  bpo::notify(vm);

  auto c = config{};
  try {
    c = config_path.extension() == ".ini" ? config::read_legacy(config_path)
                                          : config::read(config_path);
    auto const bars = utl::global_progress_bars{false};
    import(c, std::move(data_path));
  } catch (std::exception const& e) {
    fmt::println("unable to import: {}", e.what());
    fmt::println("config:\n{}", fmt::streamed(c));
    return 1;
  }

  return 0;
}

}  // namespace motis