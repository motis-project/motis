#include "motis/lookup/lookup_station_info.h"

#include "utl/to_set.h"
#include "utl/to_vec.h"

#include "motis/core/access/station_access.h"
#include "motis/core/conv/station_conv.h"

using namespace flatbuffers;

namespace motis::lookup {

Offset<LookupStationInfoResponse> lookup_station_info(
    FlatBufferBuilder& fbb, schedule const& sched,
    LookupStationInfoRequest const* req) {
  auto stations =
      utl::to_set(*req->station_ids(), [&](auto const& s) -> station const* {
        return get_station(sched, s->str());
      });

  auto const include_metas = req->include_meta_stations();
  auto const include_via_fps = req->include_stations_reachable_via_footpaths();
  for (auto expand = include_metas || include_via_fps; expand;) {
    expand = false;
    for (auto const* st : stations) {
      if (include_metas) {
        for (auto const& eq : st->equivalent_) {
          if (stations.insert(eq).second) {
            expand = true;
          }
        }
      }
      if (include_via_fps) {
        for (auto const& fp : st->outgoing_footpaths_) {
          auto const* other_station = sched.stations_.at(fp.to_station_).get();
          if (stations.insert(other_station).second) {
            expand = true;
          }
        }
      }
    }
  }

  return CreateLookupStationInfoResponse(
      fbb, fbb.CreateVector(utl::to_vec(stations, [&](station const* st) {
        return CreateLookupStationInfo(
            fbb, to_fbs(fbb, *st),
            fbb.CreateVector(utl::to_vec(
                st->external_ids_,
                [&](auto const& s) { return fbb.CreateString(s.str()); })),
            st->transfer_time_,
            fbb.CreateVector(
                utl::to_vec(st->equivalent_,
                            [&](auto const& eq) { return to_fbs(fbb, *eq); })),
            fbb.CreateVector(
                utl::to_vec(st->outgoing_footpaths_, [&](auto const& fp) {
                  return CreateLookupFootpathInfo(
                      fbb, to_fbs(fbb, *sched.stations_.at(fp.to_station_)),
                      fp.duration_);
                })));
      })));
}

}  // namespace motis::lookup
