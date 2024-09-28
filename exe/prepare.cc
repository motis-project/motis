#include <filesystem>
#include <iostream>

#include "boost/program_options.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "motis/compute_footpaths.h"
#include "motis/elevators/match_elevator.h"
#include "motis/elevators/parse_fasta.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace n = nigiri;
using namespace motis;

int main(int ac, char** av) {
  auto tt_path = fs::path{"data/tt.bin"};
  auto osr_path = fs::path{"data/osr"};
  auto out_path = fs::path{"data"};
  auto fasta_path = fs::path{"data/fasta.json"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("tt", bpo::value(&tt_path)->default_value(tt_path), "timetable path")  //
      ("osr", bpo::value(&osr_path)->default_value(osr_path), "osr data")  //
      ("fasta", bpo::value(&fasta_path)->default_value(fasta_path),
       "FaSta path")  //
      ("out,o", bpo::value(&out_path)->default_value(out_path), "output path");
  auto const pos =
      bpo::positional_options_description{}.add("tt", -1).add("osr", 1);

  auto vm = bpo::variables_map{};
  bpo::store(
      bpo::command_line_parser(ac, av).options(desc).positional(pos).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  auto e = std::error_code{};
  fs::create_directories(out_path, e);

  fmt::println("loading timetable");
  auto tt = n::timetable::read(cista::memory_holder{
      cista::file{tt_path.generic_string().c_str(), "r"}.content()});

  fmt::println("loading ways");
  auto const w = osr::ways{osr_path, cista::mmap::protection::READ};

  fmt::println("loading lookup");
  auto const l = osr::lookup{w};

  fmt::println("loading platforms");
  auto pl = osr::platforms{osr_path, cista::mmap::protection::READ};

  fmt::println("building platform rtree");
  pl.build_rtree(w);

  fmt::println("computing footpaths");
  auto const elevator_footpath_map = compute_footpaths(*tt, w, l, pl, true);

  fmt::println("writing elevator footpaths");
  write(out_path / "elevator_footpath_map.bin", elevator_footpath_map);

  fmt::println("writing timetable");
  tt->write(out_path / "tt.bin");
}