#include "fmt/core.h"

#include "boost/asio/io_context.hpp"
#include "boost/program_options.hpp"

#include "net/web_server/query_router.h"
#include "net/web_server/web_server.h"

#include "rtree.h"

#include "geo/latlng.h"

#include "cista/strong.h"

#include "net/run.h"

#include "nigiri/types.h"

#include "osr/geojson.h"

#include "icc/match.h"
#include "icc/point_rtree.h"

namespace asio = boost::asio;
namespace http = boost::beast::http;
namespace n = nigiri;
namespace fs = std::filesystem;
namespace bpo = boost::program_options;
namespace json = boost::json;

std::string get_names(osr::platforms const& pl, osr::platform_idx_t const x) {
  auto ss = std::stringstream{};
  for (auto const& y : pl.platform_names_[x]) {
    ss << y.view();
  }
  return ss.str();
}

int main(int ac, char** av) {
  auto tt_path = fs::path{"tt.bin"};
  auto osr_path = fs::path{"osr"};
  auto matching_path = fs::path{"matching.bin"};

  auto desc = bpo::options_description{"Options"};
  desc.add_options()  //
      ("help,h", "produce this help message")  //
      ("tt", bpo::value(&tt_path)->default_value(tt_path), "timetable path")  //
      ("osr", bpo::value(&osr_path)->default_value(osr_path), "osr data")  //
      ("matching,m", bpo::value(&matching_path)->default_value(matching_path),
       "matching path");
  auto const pos = bpo::positional_options_description{}
                       .add("matching", -1)
                       .add("osr", 1)
                       .add("tt", 2);

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

  // Read timetable.
  fmt::println("reading timetable");
  auto tt = n::timetable::read(cista::memory_holder{
      cista::file{tt_path.generic_string().c_str(), "r"}.content()});

  // Create location r-tree.
  fmt::println("creating r-tree");
  auto const loc_rtree = [&]() {
    auto t = icc::point_rtree<n::location_idx_t>{};
    for (auto i = n::location_idx_t{0U}; i != tt->n_locations(); ++i) {
      if (!tt->location_routes_[i].empty()) {
        t.add(tt->locations_.coordinates_[i], i);
      }
    }
    return t;
  }();

  auto ioc = asio::io_context{};
  auto s = net::web_server{ioc};

  auto qr = net::query_router{};
  qr.route("POST", "/matches", [&](json::value const& query) {
    auto const q = query.as_array();

    auto const min = geo::latlng{q[1].as_double(), q[0].as_double()};
    auto const max = geo::latlng{q[3].as_double(), q[2].as_double()};

    auto matches = json::array{};

    pl.find(min, max, [&](osr::platform_idx_t const p) {
      auto const center = icc::get_platform_center(pl, w, p);
      if (!center.has_value()) {
        return;
      }
      matches.emplace_back(json::value{
          {"type", "Feature"},
          {"properties",
           {{"type", "platform"},
            {"level", to_float(pl.get_level(w, p))},
            {"platform_names", fmt::format("{}", get_names(pl, p))}}},
          {"geometry", osr::to_point(osr::point::from_latlng(*center))}});
    });

    loc_rtree.find(min, max, [&](n::location_idx_t const l) {
      auto const pos = tt->locations_.coordinates_[l];
      auto const match = icc::get_match(*tt, pl, w, l);
      auto props = json::value{{"name", tt->locations_.names_[l].view()},
                               {"id", tt->locations_.ids_[l].view()},
                               {"type", "location"}}  //
                       .as_object();
      if (match == osr::platform_idx_t::invalid()) {
        props.emplace("level", "-");
      } else {
        std::visit(
            utl::overloaded{
                [&](osr::way_idx_t x) {
                  props.emplace("osm_way_id", to_idx(w.way_osm_idx_[x]));
                  props.emplace(
                      "level", to_float(w.r_->way_properties_[x].from_level()));
                },
                [&](osr::node_idx_t x) {
                  props.emplace("osm_node_id", to_idx(w.node_to_osm_[x]));
                  props.emplace(
                      "level",
                      to_float(w.r_->node_properties_[x].from_level()));
                }},
            osr::to_ref(pl.platform_ref_[match][0]));
      }
      matches.emplace_back(json::value{
          {"type", "Feature"},
          {"properties", props},
          {"geometry", osr::to_point(osr::point::from_latlng(pos))}});

      if (match == osr::platform_idx_t::invalid()) {
        return;
      }

      props.emplace("platform_names", fmt::format("{}", get_names(pl, match)));

      auto const center = icc::get_platform_center(pl, w, match);
      if (!center.has_value()) {
        return;
      }

      props.insert_or_assign("type", "match");
      matches.emplace_back(json::value{
          {"type", "Feature"},
          {"properties", props},
          {"geometry", osr::to_line_string(std::initializer_list<osr::point>(
                           {osr::point::from_latlng(*center),
                            osr::point::from_latlng(pos)}))}});
    });
    return json::value{{"type", "FeatureCollection"}, {"features", matches}};
  });
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