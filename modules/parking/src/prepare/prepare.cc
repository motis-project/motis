#include <iostream>
#include <map>
#include <thread>
#include <vector>

#include "boost/filesystem.hpp"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "utl/verify.h"

#include "version.h"

#include "motis/core/common/logging.h"
#include "motis/parking/prepare/foot_edges.h"
#include "motis/parking/prepare/parking.h"
#include "motis/parking/prepare/stations.h"
#include "motis/ppr/profiles.h"

namespace fs = boost::filesystem;
using namespace motis;
using namespace motis::logging;

namespace motis::parking::prepare {

struct prepare_settings : public conf::configuration {
  prepare_settings() : configuration("Prepare Options", "") {
    param(osm_, "osm", "/path/to/germany-latest.osm.pbf");
    param(parking_file_, "parking", "/path/to/parking.txt");
    param(schedule_, "schedule", "/path/to/rohdaten");
    param(footedges_db_file_, "db", "/path/to/parking_footedges.db");
    param(ppr_graph_, "ppr_graph", "/path/to/ppr-routing-graph.ppr");
    param(ppr_profiles_, "ppr_profile", "ppr search profile");
    param(edge_rtree_max_size_, "edge_rtree_max_size",
          "Maximum size for ppr edge r-tree file");
    param(area_rtree_max_size_, "area_rtree_max_size",
          "Maximum size for ppr area r-tree file");
    param(lock_rtrees_, "lock_rtrees", "Lock ppr r-trees in memory");
    param(max_walk_duration_, "max_walk_duration",
          "max walk duration (minutes)");
    param(threads_, "threads", "number of threads");
  }

  std::string osm_{"germany-latest.osm.pbf"};
  std::string parking_file_{"parking.txt"};
  std::string schedule_{"rohdaten"};
  std::string footedges_db_file_{"parking_footedges.db"};
  std::string ppr_graph_{"routing-graph.ppr"};
  std::vector<std::string> ppr_profiles_;
  std::size_t edge_rtree_max_size_{1024UL * 1024 * 1024 * 3};
  std::size_t area_rtree_max_size_{1024UL * 1024 * 1024};
  bool lock_rtrees_{false};
  int max_walk_duration_{10};
  unsigned threads_{std::thread::hardware_concurrency()};
};

}  // namespace motis::parking::prepare

int main(int argc, char const** argv) {
  using namespace motis::parking;
  using namespace motis::parking::prepare;
  prepare_settings opt;

  try {
    conf::options_parser parser({&opt});
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      std::cout << "\n\tparking-prepare (MOTIS v" << short_version() << ")\n\n";
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      std::cout << "parking-prepare (MOTIS v" << long_version() << ")\n";
      return 0;
    }

    parser.read_configuration_file(true);
    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  utl::verify(fs::is_regular_file(opt.osm_), "cannot find osm dataset");

  std::map<std::string, ::ppr::routing::search_profile> ppr_profiles;
  motis::ppr::read_profile_files(opt.ppr_profiles_, ppr_profiles);
  if (ppr_profiles.empty()) {
    std::cout << "warning: no ppr profiles specified" << std::endl;
  }
  if (ppr_profiles.find("default") == end(ppr_profiles)) {
    std::cout << "adding implicit ppr default profile" << std::endl;
    ppr_profiles["default"] = {};
  }
  for (auto& p : ppr_profiles) {
    p.second.duration_limit_ = opt.max_walk_duration_ * 60;
  }

  std::vector<motis::parking::parking_lot> parking_data;
  if (extract_parkings(opt.osm_, opt.parking_file_, parking_data)) {
    std::cout << "Parking data written to " << opt.parking_file_ << std::endl;
  } else {
    std::cout << "Parking data extraction failed" << std::endl;
    return 1;
  }

  parkings park(std::move(parking_data));
  stations st(opt.schedule_);

  std::cout << park.parkings_.size() << " parkings, " << st.size()
            << " stations" << std::endl;

  compute_foot_edges(st, park, opt.footedges_db_file_, opt.ppr_graph_,
                     opt.edge_rtree_max_size_, opt.area_rtree_max_size_,
                     opt.lock_rtrees_, ppr_profiles, opt.threads_);

  return 0;
}
