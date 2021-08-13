#include "motis/paxmon/output/journey_converter.h"

#include <optional>

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "motis/paxmon/loader/journeys/journey_access.h"

namespace motis::paxmon::output {

journey::transport const* get_journey_transport(journey const& j,
                                                std::size_t enter_stop_idx,
                                                std::size_t exit_stop_idx) {
  journey::transport const* best_match = nullptr;
  for (auto const& t : j.transports_) {
    if (t.from_ >= exit_stop_idx || t.to_ <= enter_stop_idx) {
      continue;
    }
    if (best_match == nullptr ||
        (t.from_ == enter_stop_idx && best_match->from_ != enter_stop_idx) ||
        ((t.from_ == enter_stop_idx) == (best_match->from_ == enter_stop_idx) &&
         t.to_ > best_match->to_)) {
      best_match = &t;
    }
  }
  return best_match;
}

void for_each_leg(journey const& j,
                  std::function<void(journey::stop const&, journey::stop const&,
                                     extern_trip const&,
                                     journey::transport const*)> const& trip_cb,
                  std::function<void(journey::stop const&,
                                     journey::stop const&)> const& foot_cb) {
  std::optional<std::size_t> enter_stop_idx, exit_stop_idx;
  for (auto const& [stop_idx, stop] : utl::enumerate(j.stops_)) {
    if (stop.exit_) {
      exit_stop_idx = stop_idx;
      utl::verify(enter_stop_idx.has_value(),
                  "invalid journey: exit stop without preceding enter stop");
      auto const trip_segments =
          get_journey_trip_segments(j, *enter_stop_idx, *exit_stop_idx);
      for (auto const& jts : trip_segments) {
        trip_cb(j.stops_.at(jts.from_), j.stops_.at(jts.to_),
                jts.trip_->extern_trip_,
                get_journey_transport(j, jts.from_, jts.to_));
      }
      enter_stop_idx.reset();
    }
    if (stop.enter_) {
      enter_stop_idx = stop_idx;
      if (exit_stop_idx && *exit_stop_idx != stop_idx) {
        foot_cb(j.stops_.at(*exit_stop_idx), stop);
      }
      exit_stop_idx.reset();
    }
  }
}

journey_converter::journey_converter(const std::string& output_path)
    : writer_{output_path} {
  writer_ << "id"
          << "secondary_id"
          << "leg_idx"
          << "leg_type"
          << "from"
          << "to"
          << "enter"
          << "exit"
          << "category"
          << "train_nr"
          << "passengers" << end_row;
}

void journey_converter::write_journey(const journey& j,
                                      std::uint64_t primary_id,
                                      std::uint64_t secondary_id,
                                      std::uint16_t pax) {
  auto leg_idx = 0U;
  for_each_leg(
      j,
      [&](journey::stop const& enter_stop, journey::stop const& exit_stop,
          extern_trip const& et, journey::transport const* transport) {
        writer_ << primary_id << secondary_id << ++leg_idx << "TRIP"
                << enter_stop.eva_no_ << exit_stop.eva_no_
                << enter_stop.departure_.schedule_timestamp_
                << exit_stop.arrival_.schedule_timestamp_
                << (transport != nullptr ? transport->category_name_ : "")
                << et.train_nr_ << pax << end_row;
      },
      [&](journey::stop const& walk_from_stop,
          journey::stop const& walk_to_stop) {
        writer_ << primary_id << secondary_id << ++leg_idx << "FOOT"
                << walk_from_stop.eva_no_ << walk_to_stop.eva_no_
                << walk_from_stop.departure_.schedule_timestamp_
                << walk_to_stop.arrival_.schedule_timestamp_ << "" << 0 << pax
                << end_row;
      });
}

}  // namespace motis::paxmon::output
