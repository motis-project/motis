#include "fmt/core.h"

#include "boost/asio/deadline_timer.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "utl/read_file.h"

#include "net/run.h"

#include "osr/lookup.h"

#include "icc/elevators/elevators.h"
#include "icc/endpoints/elevators.h"
#include "icc/endpoints/footpaths.h"
#include "icc/endpoints/graph.h"
#include "icc/endpoints/levels.h"
#include "icc/endpoints/matches.h"
#include "icc/endpoints/osr_routing.h"
#include "icc/endpoints/platforms.h"
#include "icc/endpoints/update_elevator.h"
#include "icc/match_platforms.h"
#include "icc/point_rtree.h"

#include "icc-api/icc-api.h"

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
  auto const fasta = utl::read_file(fasta_path.generic_string().c_str());
  if (!fasta.has_value()) {
    fmt::println("could not read fasta file {}", fasta_path);
    return 1;
  }
  auto const elevator_nodes = get_elevator_nodes(w);
  auto e = shared_elevators{w, elevator_nodes,
                            parse_fasta(std::string_view{*fasta})};

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

  fmt::println("creating matches");
  auto const matches = get_matches(*tt, pl, w);

  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};
  auto qr =
      net::query_router{}
          .route("POST", "/api/matches", ep::matches{loc_rtree, *tt, w, l, pl})
          .route("POST", "/api/elevators", ep::elevators{e, w, l})
          .route("POST", "/api/route", ep::osr_routing{w, l, e})
          .route("POST", "/api/levels", ep::levels{w, l})
          .route("POST", "/api/platforms", ep::platforms{w, l, pl})
          .route("POST", "/api/graph", ep::graph{w, l})
          .route("POST", "/api/footpaths",
                 ep::footpaths{*tt, w, l, pl, loc_rtree, matches, e})
          .route("POST", "/api/update_elevator",
                 ep::update_elevator{e, w, elevator_nodes})
          .route("GET", "/api/route",
                 [](api::Place const& v) { return api::StepInstruction{}; });

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