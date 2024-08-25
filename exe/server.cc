#include "fmt/core.h"

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/read_file.h"

#include "net/run.h"

#include "osr/lookup.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/rt_timetable.h"

#include "icc/elevators/elevators.h"
#include "icc/endpoints/elevators.h"
#include "icc/endpoints/footpaths.h"
#include "icc/endpoints/graph.h"
#include "icc/endpoints/levels.h"
#include "icc/endpoints/matches.h"
#include "icc/endpoints/osr_routing.h"
#include "icc/endpoints/platforms.h"
#include "icc/endpoints/routing.h"
#include "icc/endpoints/update_elevator.h"
#include "icc/match_platforms.h"
#include "icc/point_rtree.h"
#include "icc/tt_location_rtree.h"
#include "icc/update_rtt_td_footpaths.h"

namespace asio = boost::asio;
namespace http = boost::beast::http;
namespace n = nigiri;
namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace json = boost::json;

using namespace icc;

int main(int ac, char** av) {
  auto tt_path = fs::path{"out"};
  auto osr_path = fs::path{"osr"};
  auto fasta_path = fs::path{"fasta.json"};
  auto adr_path = fs::path{"adr.cista"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("tt", bpo::value(&tt_path)->default_value(tt_path), "timetable path")  //
      ("osr", bpo::value(&osr_path)->default_value(osr_path), "osr data")  //
      ("adr", bpo::value(&adr_path)->default_value(adr_path), "adr path")  //
      ("fasta", bpo::value(&fasta_path)->default_value(fasta_path),
       "fasta path");
  auto const pos = bpo::positional_options_description{}
                       .add("fasta", 1)
                       .add("osr", 2)
                       .add("tt", 3);

  auto vm = bpo::variables_map{};
  bpo::store(
      bpo::command_line_parser(ac, av).options(desc).positional(pos).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") != 0U) {
    std::cout << desc << "\n";
    return 0;
  }

  // Read osr.
  fmt::println("loading ways");
  auto const w = osr::ways{osr_path, cista::mmap::protection::READ};

  fmt::println("loading platforms");
  auto pl = osr::platforms{osr_path, cista::mmap::protection::READ};
  pl.build_rtree(w);

  fmt::println("building lookup");
  auto l = osr::lookup{w};

  // Read elevators.
  auto const fasta = utl::read_file(fasta_path.generic_string().c_str());
  if (!fasta.has_value()) {
    fmt::println("could not read fasta file {}", fasta_path);
    return 1;
  }
  auto const elevator_nodes = get_elevator_nodes(w);
  auto e = std::make_shared<elevators>(w, elevator_nodes,
                                       parse_fasta(std::string_view{*fasta}));

  // Read timetable.
  fmt::println("reading timetable");
  auto const elevator_footpath_map =
      read_elevator_footpath_map(tt_path / "elevator_footpath_map.bin");
  auto tt = n::timetable::read(cista::memory_holder{
      cista::file{(tt_path / "tt.bin").generic_string().c_str(), "r"}
          .content()});
  tt->locations_.resolve_timezones();

  // Create matches location_idx_t => platform_idx_t
  fmt::println("creating matches");
  auto const matches = get_matches(*tt, pl, w);

  // Create location r-tree.
  fmt::println("creating r-tree");
  auto const loc_rtree = create_location_rtree(*tt);

  // Create time-dependent footpaths.
  fmt::println("updating time-dependent footpaths");
  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  auto rtt =
      std::make_shared<n::rt_timetable>(n::rt::create_rt_timetable(*tt, today));
  icc::update_rtt_td_footpaths(w, l, pl, *tt, loc_rtree, *e,
                               *elevator_footpath_map, matches, *rtt);

  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr = net::query_router{}
                .post("/api/matches", ep::matches{loc_rtree, *tt, w, l, pl})
                .post("/api/elevators", ep::elevators{e, w, l})
                .post("/api/route", ep::osr_routing{w, l, e})
                .post("/api/levels", ep::levels{w, l})
                .post("/api/platforms", ep::platforms{w, l, pl})
                .post("/api/graph", ep::graph{w, l})
                .post("/api/footpaths",
                      ep::footpaths{*tt, w, l, pl, loc_rtree, matches, e})
                .post("/api/update_elevator",
                      ep::update_elevator{*tt, w, l, pl, loc_rtree,
                                          elevator_nodes, matches, e, rtt})
                .get("/api/v1/plan",
                     ep::routing{w, l, pl, *tt, loc_rtree, rtt, e, matches});

  qr.serve_files("ui/build");
  qr.enable_cors();
  s.on_http_request(std::move(qr));

  auto ec = boost::system::error_code{};
  s.init("0.0.0.0", "8000", ec);
  s.run();
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  std::cout << "listening on 0.0.0.0:8000\n";
  net::run(ioc)();
}