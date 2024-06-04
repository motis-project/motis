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

#include "icc/match.h"
#include "osr/geojson.h"

namespace asio = boost::asio;
namespace http = boost::beast::http;
namespace n = nigiri;
namespace fs = std::filesystem;
namespace bpo = boost::program_options;

template <typename T>
struct point_rtree {
  point_rtree() : rtree_{rtree_new()} {}

  ~point_rtree() {
    if (rtree_ != nullptr) {
      rtree_free(rtree_);
    }
  }

  point_rtree(point_rtree const&) = delete;
  point_rtree(point_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
  }

  point_rtree& operator=(point_rtree const&) = delete;
  point_rtree& operator=(point_rtree&& o) {
    if (this != &o) {
      rtree_ = o.rtree_;
      o.rtree_ = nullptr;
    }
  }

  void add(geo::latlng const& pos, T const t) {
    auto const min_corner = std::array{pos.lng(), pos.lat()};
    rtree_insert(
        rtree_, min_corner.data(), nullptr,
        reinterpret_cast<void*>(static_cast<std::size_t>(cista::to_idx(t))));
  }

  template <typename Fn>
  void find(geo::latlng const& a, geo::latlng const& b, Fn&& fn) const {
    auto const min =
        std::array{std::min(a.lng_, b.lng_), std::min(a.lat_, b.lat_)};
    auto const max =
        std::array{std::max(a.lng_, b.lng_), std::max(a.lat_, b.lat_)};
    rtree_search(
        rtree_, min.data(), max.data(),
        [](double const* /* min */, double const* /* max */, void const* item,
           void* udata) {
          (*reinterpret_cast<Fn*>(udata))(T{static_cast<cista::base_t<T>>(
              reinterpret_cast<std::size_t>(item))});
          return true;
        },
        &fn);
  }

  rtree* rtree_{nullptr};
};

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

  // Read matching.
  fmt::println("reading matching buffer");
  auto const matching_buf =
      cista::file{matching_path.generic_string().c_str(), "r"}.content();
  auto const* matching = cista::deserialize<icc::matching_t>(matching_buf);

  // Read timetable.
  fmt::println("reading timetable");
  auto tt = n::timetable::read(cista::memory_holder{
      cista::file{tt_path.generic_string().c_str(), "r"}.content()});

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

  auto qr = net::query_router{};
  qr.route("POST", "/matches", [&](boost::json::value const& query) {
    auto const q = query.as_array();
    auto matches = boost::json::array{};
    loc_rtree.find(
        {q[1].as_double(), q[0].as_double()},
        {q[3].as_double(), q[2].as_double()}, [&](n::location_idx_t const x) {
          auto const pos = tt->locations_.coordinates_[x];
          auto const match = (*matching)[x];
          auto props =
              boost::json::value{{"name", tt->locations_.names_[x].view()},
                                 {"id", tt->locations_.ids_[x].view()}}
                  .as_object();
          if (match == osr::platform_idx_t::invalid()) {
            props.emplace("level", "-");
          } else {
            std::visit(
                utl::overloaded{
                    [&](osr::way_idx_t x) {
                      props.emplace("osm_way_idx", to_idx(w.way_osm_idx_[x]));
                      props.emplace(
                          "level", to_float(w.way_properties_[x].from_level()));
                    },
                    [&](osr::node_idx_t x) {
                      props.emplace("osm_node_idx", to_idx(w.node_to_osm_[x]));
                      props.emplace(
                          "level",
                          to_float(w.node_properties_[x].from_level()));
                    }},
                osr::to_ref(pl.platform_ref_[match][0]));
          }
          matches.emplace_back(boost::json::value{
              {"type", "Feature"},
              {"properties", props},
              {"geometry", osr::to_point(osr::point::from_latlng(pos))}});
        });
    return boost::json::value{{"type", "FeatureCollection"},
                              {"features", matches}};
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