#include "motis/ris/gtfs-rt/parse_trip_update.h"

#include <algorithm>

#include "boost/date_time/gregorian/gregorian_types.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

#include "utl/verify.h"

#include "motis/core/common/date_time_util.h"
#include "motis/core/common/logging.h"
#include "motis/core/schedule/event.h"
#include "motis/core/schedule/event_type.h"
#include "motis/core/access/edge_access.h"
#include "motis/core/access/station_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/parse_event.h"
#include "motis/ris/gtfs-rt/parse_stop.h"
#include "motis/ris/gtfs-rt/parse_time.h"

#ifdef CreateEvent
#undef CreateEvent
#endif

using namespace transit_realtime;
using namespace flatbuffers;
using namespace motis::logging;

namespace motis::ris::gtfsrt {

void collect_events(trip_update_context& update_ctx,
                    std::unique_ptr<knowledge_context> const& knowledge) {
  auto const& trip = *update_ctx.trip_;
  auto skipped_stops = update_ctx.known_stop_skips_;
  auto trip_update = update_ctx.trip_update_;
  auto& sched = update_ctx.sched_;

  stop_context stop_ctx;
  for (auto const& stu : trip_update.stop_time_update()) {
    stop_ctx.update(sched, trip, stu, skipped_stops);

    if (!stu.has_schedule_relationship() ||
        stu.schedule_relationship() ==
            TripUpdate_StopTimeUpdate_ScheduleRelationship_SCHEDULED) {

      if (skipped_stops != nullptr &&
          skipped_stops->is_skipped(stop_ctx.seq_no_)) {
        throw std::runtime_error(
            "Stop " + stop_ctx.station_id_ +
            " previously reported as SKIPPED is now reported as SCHEDULED. "
            "This is not supported.");
      }

      auto const has_delay_for_addition =
          [&](TripUpdate_StopTimeEvent const& time_evt) -> bool {
        return (!update_ctx.is_addition_ ||
                (update_ctx.is_addition_ && time_evt.has_delay()));
      };

      if (stu.has_arrival() && has_delay_for_addition(stu.arrival()) &&
          stop_ctx.idx_ > 0) {
        evt event(trip, stop_ctx, event_type::ARR);
        event.new_sched_time_ = get_updated_time(
            stu.arrival(), event.orig_sched_time_, update_ctx.is_addition_);
        event.verify_times(sched);
        update_ctx.is_events_.emplace_back(event);
      }
      if (stu.has_departure() && has_delay_for_addition(stu.departure()) &&
          stop_ctx.idx_ < trip.edges_->size()) {
        evt event(trip, stop_ctx, event_type::DEP);
        event.new_sched_time_ = get_updated_time(
            stu.departure(), event.orig_sched_time_, update_ctx.is_addition_);
        event.verify_times(sched);
        update_ctx.is_events_.emplace_back(event);
      }

      if (skipped_stops != nullptr) {
        skipped_stops->skipped_stops_[stop_ctx.seq_no_] = false;
      }
    } else if (stu.schedule_relationship() ==
               TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED) {
      // check for already skipped stops
      if (skipped_stops == nullptr) {  // lazy initializer
        auto start_date =
            to_unix_time(parse_date(trip_update.trip().start_date()));
        skipped_stops =
            knowledge->remember_stop_skips(update_ctx.trip_id_, start_date);
      }

      if (!skipped_stops->is_skipped(stop_ctx.seq_no_)) {
        // stop was not skipped until now
        skipped_stops->skipped_stops_[stop_ctx.seq_no_] = true;
        evt evt_arr(trip, stop_ctx, event_type::ARR);
        evt evt_dep(trip, stop_ctx, event_type::DEP);

        update_ctx.is_stop_skip_new_[stop_ctx.idx_] = true;
        if (stop_ctx.idx_ > 0) {
          evt_arr.verify_times(sched);
          update_ctx.reroute_events_.emplace_back(evt_arr);
        }
        if (get_stop_edge_idx(stop_ctx.idx_, event_type::DEP) <
            trip.edges_->size()) {
          evt_dep.verify_times(sched);
          update_ctx.reroute_events_.emplace_back(evt_dep);
        }
      }
    } else {
      throw std::runtime_error(
          "Encountered unhandable stop time update schedule relationship!");
    }
  }
}

void collect_additional_events(
    trip_update_context& update_ctx,
    std::unique_ptr<knowledge_context> const& knowledge) {
  // additional trip updates always contain all stops
  auto trip_update = update_ctx.trip_update_;

  auto line_id = trip_update.trip().route_id();
  known_stop_skips* known_skips = nullptr;
  auto stop_count = 0;

  for (auto const& stu : trip_update.stop_time_update()) {
    if (known_skips != nullptr) {
      update_ctx.is_stop_skip_new_.resize(stop_count + 1);
    }

    evt base;
    base.line_id_ = line_id;
    base.stop_id_ = stu.stop_id();
    base.seq_no_ = stu.stop_sequence();
    // all stops need to be given to create the trip
    base.stop_idx_ = stop_count;

    if (!stu.has_schedule_relationship() ||
        stu.schedule_relationship() ==
            TripUpdate_StopTimeUpdate_ScheduleRelationship_SCHEDULED) {

      if (known_skips != nullptr && known_skips->is_skipped(base.seq_no_)) {
        throw std::runtime_error(
            "Stop " + base.stop_id_ +
            " previously reported as SKIPPED is now reported as SCHEDULED. "
            "This is not supported.");
      }

      auto parse_event = [&](TripUpdate_StopTimeEvent const& time_event,
                             event_type const type) {
        evt additional{base};
        additional.type_ = type;
        additional.orig_sched_time_ = time_event.time();
        if (update_ctx.is_new_addition_) {
          additional.verify_times(update_ctx.sched_);
          update_ctx.additional_events_.emplace_back(additional);
        }

        if (time_event.has_delay()) {
          evt delay{additional};
          delay.new_sched_time_ = delay.orig_sched_time_ + time_event.delay();
          delay.verify_times(update_ctx.sched_);
          update_ctx.is_events_.emplace_back(delay);
        }
      };

      if (stu.has_arrival()) {
        parse_event(stu.arrival(), event_type::ARR);
      }
      if (stu.has_departure()) {
        parse_event(stu.departure(), event_type::DEP);
      }
      ++stop_count;
    } else if (stu.schedule_relationship() ==
               TripUpdate_StopTimeUpdate_ScheduleRelationship_SKIPPED) {
      // only note that the sequence number of the stop is skipped for future
      // trip updates but do not report an reroute event do not increase the
      // stop count as these stops are not relevant and
      // directly a shrinked addition train is reported
      if (!update_ctx.is_addition_skip_allowed_) {
        throw std::runtime_error(
            "Found skipped stop " + base.stop_id_ +
            " on additional trip which was configured to not be present!");
      }

      if (known_skips == nullptr) {
        update_ctx.known_stop_skips_ = knowledge->remember_stop_skips(
            update_ctx.trip_id_, update_ctx.trip_start_date_);
        known_skips = update_ctx.known_stop_skips_;
      }

      known_skips->skipped_stops_[base.seq_no_] = true;
    } else {
      throw std::runtime_error(
          "Encountered unhandable stop time update schedule relationship!");
    }
  }
}

void collect_canceled_events(
    trip_update_context& update_ctx,
    std::unique_ptr<knowledge_context> const& knowledge) {

  auto& sched = update_ctx.sched_;
  auto trip_update = update_ctx.trip_update_;

  if (!update_ctx.is_new_canceled_) {
    return;
  }

  access::stops stop_it{update_ctx.trip_};
  stop_context stop_ctx;
  std::for_each(
      begin(stop_it), end(stop_it), [&](access::trip_stop const& stop) {
        stop_ctx.idx_ = stop.index();
        stop_ctx.station_id_ = stop.get_station(sched).eva_nr_;
        stop_ctx.is_skip_known_ = true;

        if (stop.index() > 0) {
          stop_ctx.stop_arrival_ = get_schedule_time(
              *update_ctx.trip_, sched, stop.index(), event_type::ARR);
          evt arr(*update_ctx.trip_, stop_ctx, event_type::ARR);
          arr.verify_times(sched);
          update_ctx.reroute_events_.emplace_back(arr);
        }

        if (stop.index() < update_ctx.trip_->edges_->size()) {
          stop_ctx.stop_departure_ = get_schedule_time(
              *update_ctx.trip_, sched, stop.index(), event_type::DEP);
          evt dep(*update_ctx.trip_, stop_ctx, event_type::DEP);
          dep.verify_times(sched);
          update_ctx.reroute_events_.emplace_back(dep);
        }
      });

  knowledge->remember_canceled(trip_update.trip());
}

void check_and_fix_reroute(trip_update_context& update_ctx) {
  // currently problems can arise if a stop is skipped but the departure for the
  // previous stop was not cancelled as well
  // the same goes for cancelled stops at the beginning of the trip which lead
  // to an arrival from nowhere intermediate skips are no problem as the
  // departure leads to an arrival afterwards
  auto& sched = update_ctx.sched_;
  auto unskipped_before_skipped = !update_ctx.is_stop_skip_new_[0];
  auto unskipped_after_skipped =
      !update_ctx.is_stop_skip_new_[update_ctx.is_stop_skip_new_.size() - 1];

  if (unskipped_after_skipped && unskipped_before_skipped) {
    return;
  }

  auto const add_event = [&](int const stop_idx, event_type const type) {
    evt event;
    event.stop_idx_ = stop_idx;
    event.type_ = type;
    event.stop_id_ = access::trip_stop{update_ctx.trip_, event.stop_idx_}
                         .get_station(sched)
                         .eva_nr_;
    event.orig_sched_time_ =
        get_schedule_time(*update_ctx.trip_, sched, event.stop_idx_, type);
    update_ctx.reroute_events_.emplace_back(event);
  };

  for (int i = 1;
       i < update_ctx.is_stop_skip_new_.size() && !unskipped_before_skipped;
       ++i) {
    if (update_ctx.is_stop_skip_new_[i - 1] &&
        !update_ctx.is_stop_skip_new_[i]) {
      add_event(i, event_type::ARR);
      unskipped_before_skipped = true;
    }
  }

  for (int i = update_ctx.is_stop_skip_new_.size() - 2;
       i >= 0 && !unskipped_after_skipped; --i) {
    if (update_ctx.is_stop_skip_new_[i + 1] &&
        !update_ctx.is_stop_skip_new_[i]) {
      add_event(i, event_type::DEP);
      unskipped_after_skipped = true;
    }
  }
}

void check_and_fix_addition(trip_update_context& update_ctx) {
  auto& add_evts = update_ctx.additional_events_;

  if (add_evts[0].type_ != event_type::DEP) {
    add_evts.erase(begin(add_evts));
  }

  if (add_evts[add_evts.size() - 1].type_ != event_type::ARR) {
    add_evts.erase(end(add_evts) - 1);
  }

  if (add_evts.size() < 2) {
    throw std::runtime_error(
        "Found additional trip with only one collected event! " +
        update_ctx.trip_id_);
  }
}

void check_and_fix_delay_with_additional(trip_update_context& update_ctx) {
  auto& delays = update_ctx.is_events_;
  auto& addition = update_ctx.additional_events_;

  for (uint64_t i = 0U; i < delays.size(); ++i) {
    auto delay_stop_idx = delays[i].stop_idx_;
    auto delay_evt_type = delays[i].type_;

    auto addition_idx =
        2 * delay_stop_idx + (delay_evt_type == event_type::ARR ? -1 : 0);

    if (addition_idx < 0 || addition_idx >= addition.size() ||
        addition[addition_idx].stop_idx_ != delay_stop_idx ||
        addition[addition_idx].type_ != delay_evt_type) {
      delays.erase(begin(delays) + i);
      --i;
    }
  }
}

void check_and_fix_delay_with_skips(trip_update_context& update_ctx) {
  auto& delays = update_ctx.is_events_;

  auto first_unskipped_stop_idx = 0;
  while (update_ctx.is_stop_skip_new_[first_unskipped_stop_idx]) {
    ++first_unskipped_stop_idx;
  }

  auto last_unskipped_stop_idx = update_ctx.is_stop_skip_new_.size() - 1;
  while (update_ctx.is_stop_skip_new_[last_unskipped_stop_idx]) {
    --last_unskipped_stop_idx;
  }

  for (uint64_t i = 0U; i < delays.size(); ++i) {
    if (delays[i].stop_idx_ == first_unskipped_stop_idx &&
        delays[i].type_ == event_type::ARR) {
      delays.erase(begin(delays) + i);
    }

    if (delays[i].stop_idx_ == last_unskipped_stop_idx &&
        delays[i].type_ == event_type::DEP) {
      delays.erase(begin(delays) + i);
      --i;
    }
  }
}

void check_and_fix_implicit_cancel(
    trip_update_context& update_ctx,
    std::unique_ptr<knowledge_context> const& knowledge) {
  auto& reroute = update_ctx.reroute_events_;
  if (reroute.size() == update_ctx.trip_->edges_->size() * 2 &&
      !update_ctx.is_canceled_) {
    update_ctx.is_canceled_ = true;
    update_ctx.is_new_canceled_ = true;
    knowledge->remember_canceled(update_ctx.trip_id_,
                                 update_ctx.trip_start_date_);
    update_ctx.is_events_.clear();
    update_ctx.additional_events_.clear();
    update_ctx.forecast_event_.clear();
  }
}

void initialize_update_context(
    std::unique_ptr<knowledge_context> const& knowledge,
    trip_update_context& update_ctx) {
  auto& sched = update_ctx.sched_;
  auto trip_update = update_ctx.trip_update_;
  update_ctx.is_addition_ = trip_update.trip().has_schedule_relationship() &&
                            trip_update.trip().schedule_relationship() ==
                                TripDescriptor_ScheduleRelationship_ADDED;
  update_ctx.is_new_addition_ =
      update_ctx.is_addition_ &&
      !knowledge->is_additional_known(trip_update.trip());

  update_ctx.is_canceled_ = trip_update.trip().has_schedule_relationship() &&
                            trip_update.trip().schedule_relationship() ==
                                TripDescriptor_ScheduleRelationship_CANCELED;

  update_ctx.is_new_canceled_ = update_ctx.is_canceled_ &&
                                !knowledge->is_cancel_known(trip_update.trip());

  update_ctx.trip_id_ = trip_update.trip().trip_id();

  update_ctx.trip_start_date_ =
      to_unix_time(parse_date(trip_update.trip().start_date()));

  if (update_ctx.is_addition_ && !update_ctx.is_new_addition_) {
    auto const prim_id =
        knowledge
            ->find_additional(update_ctx.trip_id_, update_ctx.trip_start_date_)
            .primary_id_;
    update_ctx.trip_ = find_trip(sched, prim_id);
  } else if (!update_ctx.is_addition_) {
    update_ctx.trip_ =
        get_trip(sched, update_ctx.trip_id_, update_ctx.trip_start_date_);
  } else {
    return;
  }

  utl::verify(update_ctx.trip_ != nullptr,
              "GTFS trip update: unkown trip \"{}\"", update_ctx.trip_id_);

  update_ctx.is_stop_skip_new_.resize(
      (update_ctx.is_new_addition_ ||
       (update_ctx.is_addition_ && !update_ctx.is_addition_skip_allowed_))
          ? trip_update.stop_time_update_size()
          : update_ctx.trip_->edges_->size() + 1);
  update_ctx.known_stop_skips_ =
      knowledge->find_trip_stop_skips(trip_update.trip());

  if (knowledge->is_cancel_known(trip_update.trip()) &&
      !update_ctx.is_canceled_ && !update_ctx.is_new_canceled_) {
    throw std::runtime_error(
        "Trip was previously reported as canceled but is now reported "
        "different. This is not supported!");
  }
}

void handle_trip_update(
    trip_update_context& update_ctx,
    std::unique_ptr<knowledge_context> const& knowledge,
    const std::time_t timestamp,
    std::function<void(message_context&, flatbuffers::Offset<Message>)> const&
        place_msg) {
  auto& sched = update_ctx.sched_;
  initialize_update_context(knowledge, update_ctx);

  if (update_ctx.is_new_addition_ ||
      (update_ctx.is_addition_ && !update_ctx.is_addition_skip_allowed_)) {
    collect_additional_events(update_ctx, knowledge);
  } else if (update_ctx.is_canceled_) {
    collect_canceled_events(update_ctx, knowledge);
  } else {
    collect_events(update_ctx, knowledge);
  }

  if (!update_ctx.additional_events_.empty()) {
    check_and_fix_addition(update_ctx);
    auto const& first_evt = update_ctx.additional_events_[0];
    auto const motis_start_date =
        unix_to_motistime(sched, first_evt.orig_sched_time_);
    auto const station = get_station(sched, first_evt.stop_id_);
    knowledge->remember_additional(update_ctx.trip_id_,
                                   update_ctx.trip_start_date_,
                                   motis_start_date, station->index_);

    if (!update_ctx.is_events_.empty()) {
      check_and_fix_delay_with_additional(update_ctx);
    }
  }

  if (!update_ctx.reroute_events_.empty()) {
    check_and_fix_implicit_cancel(update_ctx, knowledge);
  }

  if (!update_ctx.reroute_events_.empty() && !update_ctx.is_canceled_) {
    check_and_fix_reroute(update_ctx);

    if (update_ctx.is_addition_) {
      // need to update the primary id saved to identify this
      // additional trip
      auto first_valid_stop_idx = 0;
      while (update_ctx.is_stop_skip_new_[first_valid_stop_idx]) {
        ++first_valid_stop_idx;
      }

      auto stop = access::trip_stop{update_ctx.trip_, first_valid_stop_idx};
      knowledge->update_additional(
          update_ctx.trip_id_, update_ctx.trip_start_date_,
          stop.dep_lcon().d_time_, stop.get_station_id());
    }

    if (!update_ctx.is_events_.empty()) {
      check_and_fix_delay_with_skips(update_ctx);
    }
  }

  auto const build_id_event = [&](message_context& ctx) -> Offset<IdEvent> {
    if (update_ctx.is_new_addition_) {
      auto first_evt = update_ctx.additional_events_[0];
      return create_id_event(ctx, first_evt.stop_id_,
                             first_evt.orig_sched_time_);
    } else if (update_ctx.is_addition_ &&
               !update_ctx.is_addition_skip_allowed_) {
      auto const& prim_id = knowledge
                                ->find_additional(update_ctx.trip_id_,
                                                  update_ctx.trip_start_date_)
                                .primary_id_;
      return create_id_event(ctx, sched.stations_[prim_id.station_id_]->eva_nr_,
                             motis_to_unixtime(sched, prim_id.time_));
    } else {
      return create_id_event(ctx, sched, *update_ctx.trip_);
    }
  };

  if (!update_ctx.additional_events_.empty()) {
    message_context ctx{timestamp};
    place_msg(ctx, create_additional_msg(ctx, build_id_event(ctx),
                                         update_ctx.additional_events_));
  }

  if (!update_ctx.reroute_events_.empty()) {
    message_context ctx{timestamp};
    if (update_ctx.is_canceled_) {
      place_msg(ctx, create_cancel_msg(ctx, build_id_event(ctx),
                                       update_ctx.reroute_events_));
    } else {
      place_msg(ctx, create_reroute_msg(ctx, build_id_event(ctx),
                                        update_ctx.reroute_events_));
    }
  }

  if (!update_ctx.is_events_.empty()) {
    message_context ctx{timestamp};
    auto id_event = build_id_event(ctx);
    auto delivery = create_delay_message(ctx, id_event, update_ctx.is_events_,
                                         DelayType_Is);
    place_msg(ctx, delivery);
  }

  if (!update_ctx.forecast_event_.empty()) {
    message_context ctx{timestamp};
    place_msg(ctx, create_delay_message(ctx, build_id_event(ctx),
                                        update_ctx.forecast_event_,
                                        DelayType_Forecast));
  }
}

}  // namespace motis::ris::gtfsrt