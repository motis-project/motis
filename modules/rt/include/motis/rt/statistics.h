#pragma once

#include <iomanip>
#include <iostream>

#include "motis/rt/additional_service_builder.h"
#include "motis/rt/reroute_result.h"

#include "motis/protocol/RISMessage_generated.h"

namespace motis::rt {

struct statistics {
  friend std::ostream& operator<<(std::ostream& o, statistics const& s) {
    auto c = [&](char const* desc, unsigned number) {
      o << "  " << std::setw(22) << desc << ": " << std::setw(9) << number
        << "\n";
    };

    o << "\nmsg types:\n";
    c("delay", s.delay_msgs_);
    c("cancel", s.cancel_msgs_);
    c("additional", s.additional_msgs_);
    c("reroute", s.reroute_msgs_);
    c("track", s.track_change_msgs_);
    c("free text", s.free_text_msgs_);

    o << "\nevs:\n";
    c("total", s.total_evs_);
    c("invalid time", s.ev_invalid_time_);
    c("station not found", s.ev_station_not_found_);
    c("trip not found", s.ev_trp_not_found_);
    c("additional train event", s.additional_not_found_);  // TODO(felix) track

    o << "\ntrips:\n";
    c("total", s.trip_total_);
    c("station not found", s.trip_station_not_found_);
    c("time not found", s.trip_time_not_found_);
    c("primary not found", s.trip_primary_not_found_);
    c("primary 0 not found", s.trip_primary_0_not_found_);

    o << "\nupdates\n";
    c("total", s.total_updates_);
    c("found", s.found_updates_);
    c("sched time mismatch", s.update_mismatch_sched_time_);
    c("time diff >5min", s.diff_gt_5_);
    c("time diff >10min", s.diff_gt_10_);
    c("time diff >30min", s.diff_gt_30_);

    o << "\ndisabled routes\n";
    c("conflicts", s.conflicting_events_);
    c("moved", s.conflicting_moved_);
    c("overtake", s.route_overtake_);

    o << "\ngraph\n";
    c("propagated", s.propagated_updates_);
    c("checked", s.graph_updates_);
    c("skipped", s.propagated_updates_ - s.graph_updates_);

    o << "\nadditional services\n";
    c("total", s.additional_total_);
    c("ok", s.additional_ok_);
    c("trip id mismatch", s.additional_trip_id_);
    c("count not even", s.additional_err_count_);
    c("bad event order", s.additional_err_order_);
    c("station not found", s.additional_err_station_);
    c("bad event time", s.additional_err_time_);
    c("duplicate trip", s.additional_duplicate_trip_);

    o << "\nreroute\n";
    c("ok", s.reroute_ok_);
    c("not found", s.reroute_trip_not_found_);
    c("event count mismatch", s.reroute_event_count_mismatch_);
    c("station mismatch", s.reroute_station_mismatch_);
    c("event order mismatch", s.reroute_event_order_mismatch_);
    c("station not found", s.reroute_rule_service_not_supported_);

    o << "\ncanceled services\n";
    c("trip not found", s.canceled_trp_not_found_);

    o << "\ntrack messages\n";
    c("trip separated", s.track_separations_);

    return o;
  }

  void print() const {
    std::clog.flush();
    std::cout << *this << std::endl;
  }

  void log_sched_time_mismatch(int diff) {
    if (diff != 0) {
      ++update_mismatch_sched_time_;
    }
    if (diff > 5) {
      ++diff_gt_5_;
      if (diff > 10) {
        ++diff_gt_10_;
        if (diff > 30) {
          ++diff_gt_30_;
        }
      }
    }
  }

  void count_message(motis::ris::MessageUnion const type) {
    switch (type) {
      case motis::ris::MessageUnion_DelayMessage: ++delay_msgs_; break;
      case motis::ris::MessageUnion_CancelMessage: ++cancel_msgs_; break;
      case motis::ris::MessageUnion_AdditionMessage: ++additional_msgs_; break;
      case motis::ris::MessageUnion_RerouteMessage: ++reroute_msgs_; break;
      case motis::ris::MessageUnion_ConnectionDecisionMessage:
        ++con_decision_msgs_;
        break;
      case motis::ris::MessageUnion_ConnectionAssessmentMessage:
        ++con_assessment_msgs_;
        break;
      case motis::ris::MessageUnion_TrackMessage: ++track_change_msgs_; break;
      case motis::ris::MessageUnion_FreeTextMessage: ++free_text_msgs_; break;
      default: break;
    }
  }

  void count_additional(additional_service_builder::status const& s) {
    ++additional_total_;
    switch (s) {
      case additional_service_builder::status::OK: ++additional_ok_; break;
      case additional_service_builder::status::TRIP_ID_MISMATCH:
        ++additional_trip_id_;
        break;
      case additional_service_builder::status::EVENT_COUNT_MISMATCH:
        ++additional_err_count_;
        break;
      case additional_service_builder::status::EVENT_ORDER_MISMATCH:
        ++additional_err_order_;
        break;
      case additional_service_builder::status::STATION_NOT_FOUND:
        ++additional_err_station_;
        break;
      case additional_service_builder::status::EVENT_TIME_OUT_OF_RANGE:
        ++additional_err_time_;
        break;
      case additional_service_builder::status::DECREASING_TIME:
        ++additional_decreasing_ev_time_;
        break;
      case additional_service_builder::status::STATION_MISMATCH:
        ++additional_station_mismatch_;
        break;
      case additional_service_builder::status::DUPLICATE_TRIP:
        ++additional_duplicate_trip_;
        break;
    }
  }

  void count_reroute(reroute_result const result) {
    switch (result) {
      case reroute_result::OK: ++reroute_ok_; break;
      case reroute_result::TRIP_NOT_FOUND: ++reroute_trip_not_found_; break;
      case reroute_result::EVENT_COUNT_MISMATCH:
        ++reroute_event_count_mismatch_;
        break;
      case reroute_result::STATION_MISMATCH: ++reroute_station_mismatch_; break;
      case reroute_result::EVENT_ORDER_MISMATCH:
        ++reroute_event_order_mismatch_;
        break;
      case reroute_result::RULE_SERVICE_REROUTE_NOT_SUPPORTED:
        ++reroute_rule_service_not_supported_;
        break;
    }
  }

  bool sanity_check_fails() const {
    return (ev_invalid_time_ + ev_station_not_found_ + ev_trp_not_found_ +
            additional_not_found_ + unresolved_events_ +
            update_time_out_of_schedule_ + trip_station_not_found_ +
            trip_time_not_found_ + trip_primary_not_found_ +
            trip_primary_0_not_found_ + reroute_trip_not_found_ +
            reroute_event_count_mismatch_ + reroute_station_mismatch_ +
            reroute_event_order_mismatch_ +
            reroute_rule_service_not_supported_ + additional_err_count_ +
            additional_err_order_ + additional_err_station_ +
            additional_err_time_ + additional_decreasing_ev_time_ +
            additional_station_mismatch_ + additional_duplicate_trip_ +
            canceled_trp_not_found_) != 0;
  }

  unsigned delay_msgs_ = 0;
  unsigned cancel_msgs_ = 0;
  unsigned additional_msgs_ = 0;
  unsigned reroute_msgs_ = 0;
  unsigned con_decision_msgs_ = 0;
  unsigned con_assessment_msgs_ = 0;
  unsigned track_change_msgs_ = 0;
  unsigned free_text_msgs_ = 0;

  unsigned total_evs_ = 0;
  unsigned ev_invalid_time_ = 0;
  unsigned ev_station_not_found_ = 0;
  unsigned ev_trp_not_found_ = 0;
  unsigned additional_not_found_ = 0;
  unsigned unresolved_events_ = 0;
  unsigned update_time_out_of_schedule_ = 0;

  unsigned trip_total_ = 0;
  unsigned trip_station_not_found_ = 0;
  unsigned trip_time_not_found_ = 0;
  unsigned trip_primary_not_found_ = 0;
  unsigned trip_primary_0_not_found_ = 0;

  unsigned total_updates_ = 0;
  unsigned found_updates_ = 0;
  unsigned update_mismatch_sched_time_ = 0;
  unsigned diff_gt_5_ = 0, diff_gt_10_ = 0, diff_gt_30_ = 0;

  unsigned conflicting_events_ = 0;
  unsigned conflicting_moved_ = 0;
  unsigned route_overtake_ = 0;

  unsigned reroute_ok_ = 0;
  unsigned reroute_trip_not_found_ = 0;
  unsigned reroute_event_count_mismatch_ = 0;
  unsigned reroute_station_mismatch_ = 0;
  unsigned reroute_event_order_mismatch_ = 0;
  unsigned reroute_rule_service_not_supported_ = 0;

  unsigned propagated_updates_ = 0;
  unsigned graph_updates_ = 0;

  unsigned additional_total_ = 0;
  unsigned additional_ok_ = 0;
  unsigned additional_trip_id_ = 0;
  unsigned additional_err_count_ = 0;
  unsigned additional_err_order_ = 0;
  unsigned additional_err_station_ = 0;
  unsigned additional_err_time_ = 0;
  unsigned additional_decreasing_ev_time_ = 0;
  unsigned additional_station_mismatch_ = 0;
  unsigned additional_duplicate_trip_ = 0;

  unsigned canceled_trp_not_found_ = 0;

  unsigned track_separations_ = 0;
};

}  // namespace motis::rt
