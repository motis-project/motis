#include "motis/parking/prepare/stations.h"

#include "boost/filesystem.hpp"

#include "utl/parser/file.h"

#include "utl/to_vec.h"

#include "motis/core/common/logging.h"
#include "motis/parking/prepare/use_64bit_flatbuffers.h"
#include "motis/schedule-format/Schedule_generated.h"

namespace fs = boost::filesystem;
using namespace motis::loader;
using namespace motis::logging;

namespace motis::parking::prepare {

std::vector<station> load_stations(std::string const& schedule_path) {
  scoped_timer timer("Loading stations");
  auto const sched_file = fs::path(schedule_path) / "schedule.raw";
  if (!fs::is_regular_file(sched_file)) {
    throw std::runtime_error("cannot open schedule.raw");
  }

  auto const buf = utl::file(sched_file.string().c_str(), "r").content();
  auto const sched = GetSchedule(buf.buf_);

  return utl::to_vec(*sched->stations(), [](Station const* st) {
    return station{st->id()->str(), geo::latlng{st->lat(), st->lng()}};
  });
}

stations::stations(std::string const& schedule_path) {
  stations_ = load_stations(schedule_path);
  geo_index_ =
      geo::make_point_rtree(stations_, [](auto const& s) { return s.pos_; });
}

std::vector<std::pair<station, double>> stations::get_in_radius(
    geo::latlng const& center, double radius) const {
  return utl::to_vec(
      geo_index_.in_radius_with_distance(center, radius), [&](auto const& s) {
        return std::make_pair(stations_[std::get<1>(s)], std::get<0>(s));
      });
}

}  // namespace motis::parking::prepare
