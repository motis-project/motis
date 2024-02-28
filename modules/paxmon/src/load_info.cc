#include "motis/paxmon/load_info.h"

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/core/access/realtime_access.h"

#include "motis/paxmon/get_load.h"
#include "motis/paxmon/trip_section_load_iterator.h"

namespace motis::paxmon {

inline bool get_possibly_over_capacity(edge const* e, pax_cdf const& cdf) {
  return e->has_capacity() && load_factor_possibly_ge(cdf, e->capacity(), 1.F);
}

inline bool get_possibly_over_capacity(trip_section_with_load const& sec,
                                       pax_cdf const& cdf) {
  return sec.has_capacity_info() &&
         load_factor_possibly_ge(cdf, sec.capacity(), 1.F);
}

inline float get_probability_over_capacity(edge const* e, pax_cdf const& cdf) {
  return e->has_capacity() ? get_pax_gt_probability(cdf, e->capacity()) : 0.F;
}

inline float get_probability_over_capacity(trip_section_with_load const& sec,
                                           pax_cdf const& cdf) {
  return sec.has_capacity_info() ? get_pax_gt_probability(cdf, sec.capacity())
                                 : 0.F;
}

inline basic_event_info get_basic_event_info(event_node const* evn) {
  return basic_event_info{.station_idx_ = evn->station_idx(),
                          .schedule_time_ = evn->schedule_time(),
                          .current_time_ = evn->current_time()};
}

inline basic_event_info get_basic_event_info(schedule const& sched,
                                             ev_key const ek) {
  return basic_event_info{.station_idx_ = ek.get_station_idx(),
                          .schedule_time_ = get_schedule_time(sched, ek),
                          .current_time_ = ek.get_time()};
}

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf&& pdf, pax_cdf&& cdf, bool updated) {
  auto const possibly_over_capacity = get_possibly_over_capacity(e, cdf);
  auto const probability_over_capacity = get_probability_over_capacity(e, cdf);
  auto const expected_pax = get_expected_load(uv, e->pci_);
  return edge_load_info{.forecast_pdf_ = std::move(pdf),
                        .forecast_cdf_ = std::move(cdf),
                        .updated_ = updated,
                        .possibly_over_capacity_ = possibly_over_capacity,
                        .probability_over_capacity_ = probability_over_capacity,
                        .expected_passengers_ = expected_pax,
                        .capacity_ = e->capacity(),
                        .capacity_source_ = e->get_capacity_source(),
                        .from_ = get_basic_event_info(e->from(uv)),
                        .to_ = get_basic_event_info(e->to(uv))};
}

edge_load_info make_edge_load_info(universe const& uv, edge const* e,
                                   pax_pdf const& pdf, pax_cdf const& cdf,
                                   bool updated) {
  auto const possibly_over_capacity = get_possibly_over_capacity(e, cdf);
  auto const probability_over_capacity = get_probability_over_capacity(e, cdf);
  auto const expected_pax = get_expected_load(uv, e->pci_);
  return edge_load_info{.forecast_pdf_ = pdf,
                        .forecast_cdf_ = cdf,
                        .updated_ = updated,
                        .possibly_over_capacity_ = possibly_over_capacity,
                        .probability_over_capacity_ = probability_over_capacity,
                        .expected_passengers_ = expected_pax,
                        .capacity_ = e->capacity(),
                        .capacity_source_ = e->get_capacity_source(),
                        .from_ = get_basic_event_info(e->from(uv)),
                        .to_ = get_basic_event_info(e->to(uv))};
}

edge_load_info make_edge_load_info(trip_section_with_load const& sec) {
  auto pdf = sec.load_pdf();
  auto cdf = get_cdf(pdf);
  auto const possibly_over_capacity = get_possibly_over_capacity(sec, cdf);
  auto const probability_over_capacity =
      get_probability_over_capacity(sec, cdf);

  return edge_load_info{
      .forecast_pdf_ = std::move(pdf),
      .forecast_cdf_ = std::move(cdf),
      .updated_ = false,
      .possibly_over_capacity_ = possibly_over_capacity,
      .probability_over_capacity_ = probability_over_capacity,
      .expected_passengers_ = sec.expected_load(),
      .capacity_ = sec.capacity(),
      .capacity_source_ = sec.get_capacity_source(),
      .from_ = get_basic_event_info(sec.sched_, sec.ev_key_from()),
      .to_ = get_basic_event_info(sec.sched_, sec.ev_key_to())};
}

trip_load_info calc_trip_load_info(universe const& uv, schedule const& sched,
                                   trip const* trp) {
  return trip_load_info{
      .trp_ = trp,
      .edges_ = utl::to_vec(
          sections_with_load{sched, uv, trp},
          [&](auto const& sec) { return make_edge_load_info(sec); })};
}

trip_load_info calc_trip_load_info(universe const& uv, trip const* trp) {
  return trip_load_info{
      trp,
      utl::all(uv.trip_data_.edges(trp))  //
          | utl::transform([&](auto const e) { return e.get(uv); })  //
          | utl::remove_if([](auto const* e) { return !e->is_trip(); })  //
          | utl::transform([&](auto const* e) {
              auto pdf =
                  get_load_pdf(uv.passenger_groups_,
                               uv.pax_connection_info_.group_routes(e->pci_));
              auto cdf = get_cdf(pdf);
              return make_edge_load_info(uv, e, std::move(pdf), std::move(cdf),
                                         false);
            })  //
          | utl::vec()};
}

}  // namespace motis::paxmon
