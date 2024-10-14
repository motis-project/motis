#include <filesystem>

#include "utl/progress_tracker.h"

#include "boost/program_options.hpp"

#include "motis/config.h"
#include "motis/import.h"

namespace bpo = boost::program_options;
namespace fs = std::filesystem;

namespace motis {

int import(fs::path const& data_path, fs::path const& config_path) {
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