#include <filesystem>
#include <iostream>

#include "boost/program_options.hpp"

#include "nigiri/timetable.h"

#include "osr/lookup.h"
#include "osr/platforms.h"
#include "osr/routing/route.h"
#include "osr/ways.h"

#include "icc/compute_footpaths.h"
#include "icc/match_elevator.h"
#include "icc/parse_fasta.h"
#include "utl/parallel_for.h"

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

  fmt::println("building rtree");
  pl.build_rtree(w);

  fmt::println("reading elevators");
  auto const file = cista::mmap{fasta_path.generic_string().c_str(),
                                cista::mmap::protection::READ};
  auto const elevators = parse_fasta(file.view());

  fmt::println("creating elevators rtree");
  auto const elevators_rtree = [&]() {
    auto t = point_rtree<elevator_idx_t>{};
    for (auto const& [i, e] : utl::enumerate(elevators)) {
      t.add(e.pos_, elevator_idx_t{i});
    }
    return t;
  }();

  fmt::println("mapping elevators");
  auto const elevator_nodes = [&]() {
    auto nodes = osr::hash_set<osr::node_idx_t>{};
    for (auto way = osr::way_idx_t{0U}; way != w.n_ways(); ++way) {
      for (auto const n : w.r_->way_nodes_[way]) {
        if (w.r_->node_properties_[n].is_elevator()) {
          nodes.emplace(n);
        }
      }
    }
    return nodes;
  }();
  auto inactive = osr::hash_set<osr::node_idx_t>{};
  auto inactive_mutex = std::mutex{};
  utl::parallel_for(elevator_nodes, [&](osr::node_idx_t const n) {
    auto const e = match_elevator(elevators_rtree, elevators, w, n);
    if (e != elevator_idx_t::invalid() &&
        elevators[e].status_ == icc::status::kInactive) {
      auto const lock = std::scoped_lock{inactive_mutex};
      inactive.emplace(n);
    }
  });
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  blocked.resize(w.n_nodes());
  for (auto const n : inactive) {
    blocked.set(n, true);
  }

  fmt::println("computing footpaths");
  compute_footpaths(*tt, w, l, pl, blocked);

  fmt::println("writing result");
  tt->write(out_path);
}