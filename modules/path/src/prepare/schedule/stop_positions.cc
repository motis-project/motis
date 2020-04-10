#include "motis/path/prepare/schedule/stop_positions.h"

#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "geo/latlng.h"
#include "geo/point_rtree.h"
#include "utl/parser/file.h"
#include "utl/to_vec.h"
#include "utl/zip.h"

#include "motis/core/common/logging.h"
#include "motis/path/prepare/osm_util.h"

#include "motis/path/fbs/StopPositionCache_generated.h"

using namespace flatbuffers;
namespace fs = boost::filesystem;

namespace motis::path {

constexpr auto kStopPositionsCacheFile = "pathcache.stop_positions.raw";

bool is_stop_positions_cache_available(std::string const& osm_file,
                                       std::string const& sched_path) {
  if (!fs::is_regular_file(kStopPositionsCacheFile)) {
    return false;
  }

  auto const buf = utl::file{kStopPositionsCacheFile, "r"}.content();
  auto const* cache = GetStopPositionCache(buf.buf_);

  auto sched_file = fs::path(sched_path) / "schedule.raw";
  return cache->osm_file()->mod_date() == fs::last_write_time(osm_file) &&
         cache->osm_file()->file_size() == fs::file_size(osm_file) &&
         cache->sched_file()->mod_date() == fs::last_write_time(sched_file) &&
         cache->sched_file()->file_size() == fs::file_size(sched_file);
}

void prepare_stop_positions_cache(std::string const& osm_file,
                                  std::string const& sched_path,
                                  std::vector<station> const& stations) {
  if (is_stop_positions_cache_available(osm_file, sched_path)) {
    LOG(motis::logging::info) << "using existing stop positions";
    return;
  }
  motis::logging::scoped_timer t{"parsing stop positions"};

  utl::verify(std::is_sorted(
                  begin(stations), end(stations),
                  [](auto const& a, auto const& b) { return a.id_ < b.id_; }),
              "stations vector is not sorted by station id");

  auto const rtree =
      geo::make_point_rtree(stations, [](auto const& s) { return s.pos_; });

  std::vector<std::vector<geo::latlng>> positions;
  positions.resize(stations.size());

  std::string const stop_position = "stop_position";
  std::string const yes = "yes";
  foreach_osm_node(osm_file, [&](auto const& node) {
    if (stop_position != node.get_value_by_key("public_transport", "") ||
        yes != node.get_value_by_key("bus", "")) {
      return;
    }

    auto const pos = geo::latlng{node.location().lat(), node.location().lon()};
    for (auto const& idx : rtree.in_radius(pos, 100)) {
      positions.at(idx).emplace_back(pos);
      break;  // XXX ?
    }
  });

  FlatBufferBuilder fbb;

  std::vector<Offset<StopPositionInfo>> infos;
  infos.reserve(stations.size());
  for (auto const& [station, pos] : utl::zip(stations, positions)) {
    infos.emplace_back(CreateStopPositionInfo(
        fbb, fbb.CreateString(station.id_),
        fbb.CreateVectorOfStructs(utl::to_vec(
            pos, [](auto const& p) { return Position(p.lat_, p.lng_); }))));
  }

  auto sched_file = fs::path(sched_path) / "schedule.raw";
  fbb.Finish(CreateStopPositionCache(
      fbb,
      CreateFileID(fbb, fs::last_write_time(osm_file), fs::file_size(osm_file)),
      CreateFileID(fbb, fs::last_write_time(sched_file),
                   fs::file_size(sched_file)),
      fbb.CreateVector(infos)));
  utl::file{kStopPositionsCacheFile, "w+"}.write(fbb.GetBufferPointer(),
                                                 fbb.GetSize());
}

void annotate_stop_positions(std::vector<station>& stations) {
  auto const buf = utl::file{kStopPositionsCacheFile, "r"}.content();
  auto const* cache = GetStopPositionCache(buf.buf_);

  for (auto& station : stations) {
    auto const it = std::lower_bound(
        cache->stop_positions()->begin(), cache->stop_positions()->end(),
        station.id_, [](auto const& lhs, auto const& rhs) {
          return std::strcmp(lhs->station_id()->c_str(), rhs.c_str()) < 0;
        });

    utl::verify(it != cache->stop_positions()->end() &&
                    station.id_ == it->station_id()->c_str(),
                "annotate_stop_positions: station not found!");

    station.stop_positions_ =
        utl::to_vec(*it->stop_positions(), [](auto const& p) {
          return geo::latlng{p->lat(), p->lng()};
        });
  }
}

void find_stop_positions(std::string const& osm_file,
                         std::string const& sched_path,
                         std::vector<station>& stations) {
  prepare_stop_positions_cache(osm_file, sched_path, stations);
  annotate_stop_positions(stations);
}

}  // namespace motis::path
