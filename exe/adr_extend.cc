#include <filesystem>

#include "boost/program_options.hpp"

#include "utl/timer.h"

#include "nigiri/timetable.h"

#include "adr/adr.h"
#include "adr/area_database.h"
#include "adr/typeahead.h"

#include "motis/adr_extend_tt.h"

namespace bpo = boost::program_options;
namespace fs = std::filesystem;
namespace n = nigiri;
using namespace motis;

int main(int ac, char** av) {
  auto data_dir = fs::path{"data"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("data,d", bpo::value(&data_dir)->default_value(data_dir), "data path");

  auto const pos = bpo::positional_options_description{}.add("data", -1);

  auto vm = bpo::variables_map{};
  bpo::store(
      bpo::command_line_parser(ac, av).options(desc).positional(pos).run(), vm);
  bpo::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << '\n';
    return 0;
  }

  // Load data.
  auto t = adr::read(data_dir / "adr" / "t.bin", false);
  auto const area_db =
      adr::area_database{data_dir / "adr", cista::mmap::protection::READ};
  auto const tt = n::timetable::read(cista::memory_holder{
      cista::file{(data_dir / "tt.bin").generic_string().c_str(), "r"}
          .content()});

  // Add locations from timetable to adr typeahead.
  adr_extend_tt(*tt, area_db, *t);

  {  // Write to disk.
    auto const timer = utl::scoped_timer{"write typeahead"};
    auto mmap = cista::buf{
        cista::mmap{(data_dir / "adr" / "t.bin").generic_string().c_str(),
                    cista::mmap::protection::WRITE}};
    cista::serialize<cista::mode::WITH_STATIC_VERSION>(mmap, *t);
  }
}