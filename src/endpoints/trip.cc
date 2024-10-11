#include "motis/endpoints/trip.h"

#include "nigiri/routing/journey.h"
#include "nigiri/rt/frun.h"
#include "nigiri/timetable.h"

#include "motis/data.h"
#include "motis/journey_to_response.h"
#include "motis/parse_location.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;

namespace motis::ep {

n::rt::run resolve_run(n::timetable const& tt,
                       date::sys_days const day,
                       n::source_idx_t const src,
                       std::string_view trip_id) {
  auto const day_idx = static_cast<int>(to_idx(tt.day_idx(day)));

  auto const lb = std::lower_bound(
      begin(tt.trip_id_to_idx_), end(tt.trip_id_to_idx_), trip_id,
      [&](n::pair<n::trip_id_idx_t, n::trip_idx_t> const& a,
          n::string const& b) {
        return std::tuple(tt.trip_id_src_[a.first],
                          tt.trip_id_strings_[a.first].view()) <
               std::tuple(src, std::string_view{b});
      });

  auto const id_matches = [src, trip_id = trip_id,
                           &tt](n::trip_id_idx_t const t_id_idx) {
    return tt.trip_id_src_[t_id_idx] == src &&
           tt.trip_id_strings_[t_id_idx].view() == trip_id;
  };

  for (auto i = lb; i != end(tt.trip_id_to_idx_) && id_matches(i->first); ++i) {
    for (auto const [t_idx, stop_range] :
         tt.trip_transport_ranges_[i->second]) {
      auto const day_offset =
          tt.event_mam(t_idx, stop_range.from_, n::event_type::kDep).days();
      auto const first_dep_day = day_idx - day_offset;
      auto const t = n::transport{t_idx, n::day_idx_t{first_dep_day}};

      auto const& traffic_days =
          tt.bitfields_[tt.transport_traffic_days_[t_idx]];
      if (!traffic_days.test(static_cast<std::size_t>(first_dep_day))) {
        continue;
      }

      return n::rt::run{.t_ = t, .stop_range_ = stop_range};
    }
  }

  return {};
}

api::Itinerary trip::operator()(boost::urls::url_view const& url) const {
  auto const rt = rt_;
  auto const rtt = rt->rtt_.get();

  auto const query = api::trip_params{url.params()};
  auto const day = parse_iso_date(query.date_);
  auto const [tag, id] = split_tag_id(query.tripId_);
  auto const r = resolve_run(tt_, day, tags_.get_src(tag), id);

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
      w_, l_, tt_, tags_, pl_, nullptr, rtt, matches_, nullptr, false,
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
      from_l.get_location_idx(), to_l.get_location_idx(), cache, blocked);
}

}  // namespace motis::ep