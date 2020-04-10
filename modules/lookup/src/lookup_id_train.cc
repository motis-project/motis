#include "motis/lookup/lookup_id_train.h"

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/journeys_to_message.h"
#include "motis/lookup/util.h"

using namespace flatbuffers;
using namespace motis::routing;

namespace motis::lookup {

Offset<Connection> lookup_id_train(FlatBufferBuilder& fbb,
                                   schedule const& sched, TripId const* t) {
  auto trp = get_trip(sched, t->station_id()->str(), t->train_nr(), t->time(),
                      t->target_station_id()->str(), t->target_time(),
                      t->line_id()->str());

  journey j;
  for (auto const& s : access::stops(trp)) {
    journey::stop stop;
    stop.enter_ = false;
    stop.exit_ = false;

    auto const& station = s.get_station(sched);
    stop.eva_no_ = station.eva_nr_;
    stop.name_ = station.name_;
    stop.lat_ = station.lat();
    stop.lng_ = station.lng();

    journey::stop::event_info arr{
        false, INVALID_TIME, INVALID_TIME, timestamp_reason::SCHEDULE, "", ""};
    arr.valid_ = s.has_arrival();
    if (arr.valid_) {
      auto const& lcon = s.arr_lcon();
      arr.timestamp_ = motis_to_unixtime(sched, lcon.a_time_);
      arr.schedule_timestamp_ =
          arr.timestamp_;  // TODO(Sebastian Fahnenschreiber) get sched time
      arr.timestamp_reason_ = timestamp_reason::SCHEDULE;
      arr.track_ = sched.tracks_[lcon.full_con_->a_track_];
    }
    stop.arrival_ = arr;

    journey::stop::event_info dep{
        false, INVALID_TIME, INVALID_TIME, timestamp_reason::SCHEDULE, "", ""};
    dep.valid_ = s.has_departure();
    if (dep.valid_) {
      auto const& lcon = s.dep_lcon();
      dep.timestamp_ = motis_to_unixtime(sched, lcon.d_time_);
      dep.schedule_timestamp_ =
          dep.timestamp_;  // TODO(Sebastian Fahnenschreiber) get sched time
      dep.timestamp_reason_ = timestamp_reason::SCHEDULE;
      dep.track_ = sched.tracks_[lcon.full_con_->d_track_];
    }
    stop.departure_ = dep;

    j.stops_.push_back(stop);
  }
  j.db_costs_ = 0;
  j.night_penalty_ = 0;

  // TODO(Sebastian Fahnenschreiber) write transport
  // (using the section based iterator)
  return to_connection(fbb, j);
}

}  // namespace motis::lookup
