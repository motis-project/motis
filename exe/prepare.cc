#include <filesystem>
#include <iostream>

#include "boost/program_options.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "icc/compute_footpaths.h"
#include "icc/elevators/match_elevator.h"
#include "icc/elevators/parse_fasta.h"

namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace n = nigiri;
using namespace icc;

int main(int ac, char** av) {
  auto tt_path = fs::path{"tt.bin"};
  auto osr_path = fs::path{"osr"};
  auto out_path = fs::path{"tt_out.bin"};
  auto fasta_path = fs::path{"fasta.json"};

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

  fmt::println("reading elevators");
  auto const elevators = parse_fasta(fasta_path);

  fmt::println("creating elevators rtree");
  auto const elevators_rtree = create_elevator_rtree(elevators);

  fmt::println("mapping elevators");
  auto const elevator_nodes = get_elevator_nodes(w);
  auto const blocked =
      get_blocked_elevators(w, elevators, elevators_rtree, elevator_nodes);

  fmt::println("computing footpaths");
  compute_footpaths(*tt, w, l, pl, blocked, true);

  fmt::println("writing result");
  tt->write(out_path);
}