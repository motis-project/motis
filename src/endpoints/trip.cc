#include "motis/endpoints/trip.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/timetable.h"

#include "motis/data.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis::ep {

api::Itinerary trip::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();

  auto query = api::trip_params{url.params()};
  auto const [tag, id] = split_tag_id(query.tripId_);
  auto const src = tags_.get_src(tag);

  // transform YYYY-MM-DD to YYYYMMDD
  query.date_.erase(std::remove(begin(query.date_), end(query.date_), '-'),
                    end(query.date_));

  transit_realtime::TripDescriptor td;
  td.set_trip_id(id);
  td.set_start_date(query.date_);

  auto const [r, _] = n::rt::gtfsrt_resolve_run({}, tt_, rtt, src, td);
  utl::verify(r.valid(), "trip not found: tripId={}, date={}, tt={}",
              query.tripId_, query.date_, tt_.external_interval());

  auto fr = n::rt::frun{tt_, rtt, r};
  fr.stop_range_.to_ = fr.size();
  fr.stop_range_.from_ = 0U;
  auto const from_l = fr[0];
  auto const to_l = fr[fr.size() - 1U];
  auto const start_time = from_l.time(n::event_type::kDep);
  auto const dest_time = to_l.time(n::event_type::kArr);
  auto cache = street_routing_cache_t{};
  auto blocked = osr::bitvec<osr::node_idx_t>{};

  return journey_to_response(
      w_, l_, pl_, tt_, tags_, nullptr, rtt, matches_, shapes_, false,
      {.legs_ = {n::routing::journey::leg{
           n::direction::kForward, from_l.get_location_idx(),
           to_l.get_location_idx(), start_time, dest_time,
           n::routing::journey::run_enter_exit{
               fr,  // NOLINT(cppcoreguidelines-slicing)
               fr.first_valid(), fr.last_valid()}}},
       .start_time_ = start_time,
       .dest_time_ = dest_time,
       .dest_ = to_l.get_location_idx(),
       .transfers_ = 0U},
      tt_location{from_l.get_location_idx(),
                  from_l.get_scheduled_location_idx()},
      tt_location{to_l.get_location_idx()}, cache, blocked);
}

}  // namespace motis::ep