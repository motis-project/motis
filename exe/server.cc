#include "fmt/core.h"

#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "net/run.h"

#include "osr/lookup.h"

#include "icc/endpoints/elevators.h"
#include "icc/endpoints/matches.h"
#include "icc/endpoints/osr_routing.h"
#include "icc/match_elevator.h"
#include "icc/parse_fasta.h"
#include "icc/point_rtree.h"

namespace asio = boost::asio;
namespace http = boost::beast::http;
namespace n = nigiri;
namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace json = boost::json;

using namespace icc;

int main(int ac, char** av) {
  auto tt_path = fs::path{"tt.bin"};
  auto osr_path = fs::path{"osr"};
  auto fasta_path = fs::path{"fasta.json"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("tt", bpo::value(&tt_path)->default_value(tt_path), "timetable path")  //
      ("osr", bpo::value(&osr_path)->default_value(osr_path), "osr data")  //
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

  // Read timetable.
  fmt::println("reading timetable");
  auto tt = n::timetable::read(cista::memory_holder{
      cista::file{tt_path.generic_string().c_str(), "r"}.content()});

  // Read elevators.
  fmt::println("reading elevators");
  auto const elevators = parse_fasta(fasta_path);

  fmt::println("creating elevators rtree");
  auto const elevators_rtree = create_elevator_rtree(elevators);

  fmt::println("mapping elevators");
  auto const elevator_nodes = get_elevator_nodes(w);
  auto const blocked = std::make_shared<osr::bitvec<osr::node_idx_t>>(
      get_blocked_elevators(w, elevators, elevators_rtree, elevator_nodes));

  // Create location r-tree.
  fmt::println("creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt->n_locations(); ++i) {
      if (!tt->location_routes_[i].empty()) {
        t.add(tt->locations_.coordinates_[i], i);
      }
    }
    return t;
  }();

  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr =
      net::query_router{}
          .route("POST", "/api/matches", ep::matches{loc_rtree, *tt, w, l, pl})
          .route("POST", "/api/elevators",
                 ep::elevators{elevators_rtree, elevators, w, l})
          .route("POST", "/api/route", ep::osr_routing{w, l, blocked});

  qr.serve_files("ui/build");
  qr.enable_cors();
  s.on_http_request(std::move(qr));

  auto ec = boost::system::error_code{};
  s.init("0.0.0.0", "8080", ec);
  s.run();
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  std::cout << "listening on 0.0.0.0:8080\n";
  net::run(ioc)();
}