#include "motis/hashes.h"

#include <fstream>
#include <ostream>

#include "boost/json.hpp"

#include "cista/mmap.h"

namespace fs = std::filesystem;

namespace motis {

std::string to_str(meta_t const& h) {
  return boost::json::serialize(boost::json::value_from(h));
}

meta_t read_hashes(fs::path const& data_path, std::string const& name) {
  auto const p = (data_path / "meta" / (name + ".json"));
  if (!exists(p)) {
    return {};
  }
  auto const mmap =
      cista::mmap{p.generic_string().c_str(), cista::mmap::protection::READ};
  return boost::json::value_to<meta_t>(boost::json::parse(mmap.view()));
}

void write_hashes(fs::path const& data_path,
                  std::string const& name,
                  meta_t const& h) {
  auto const p = (data_path / "meta" / (name + ".json"));
  std::ofstream{p} << to_str(h);
}

}  // namespace motis