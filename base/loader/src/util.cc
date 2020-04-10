#include "motis/loader/util.h"

#include <iomanip>
#include <sstream>

using namespace flatbuffers64;
using namespace utl;
namespace fs = boost::filesystem;

namespace motis::loader {

std::string pad_to_7_digits(int eva_num) {
  std::stringstream s;
  s << std::setw(7) << std::setfill('0') << eva_num;
  return s.str();
}

void write_schedule(FlatBufferBuilder& b, fs::path const& path) {
  file f(path.string().c_str(), "w+");
  f.write(b.GetBufferPointer(), b.GetSize());
}

void collect_files(fs::path const& root, std::vector<fs::path>& files) {
  if (!fs::is_directory(root)) {
    files.emplace_back(root);
  } else {
    for (auto const& entry : fs::recursive_directory_iterator(root)) {
      if (fs::is_regular(entry.path())) {
        files.push_back(entry.path());
      }
    }
  }
}

}  // namespace motis::loader
