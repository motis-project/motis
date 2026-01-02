#include "motis/endpoints/update_elevator.h"

#include "net/not_found_exception.h"

#include "nigiri/rt/create_rt_timetable.h"

#include "motis/constants.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/get_loc.h"
#include "motis/railviz.h"
#include "motis/update_rtt_td_footpaths.h"

namespace json = boost::json;
namespace n = nigiri;

namespace motis::ep {

json::value update_elevator::operator()(json::value const& query) const {
  auto const& q = query.as_object();
  auto const id = q.at("id").to_number<std::int64_t>();
  auto const new_status = q.at("status").as_string() != "INACTIVE";
  auto const new_out_of_service = parse_out_of_service(q);

  auto const rt_copy = rt_;
  auto const e = rt_copy->e_.get();
  utl::verify<net::not_found_exception>(e != nullptr,
                                        "elevators not available");

  auto const rtt = rt_copy->rtt_.get();
  auto elevators_copy = e->elevators_;
  auto const it =
      utl::find_if(elevators_copy, [&](auto&& x) { return x.id_ == id; });
  utl::verify<net::not_found_exception>(it != end(elevators_copy),
                                        "id {} not found", id);

  it->status_ = new_status;
  it->out_of_service_ = new_out_of_service;
  it->state_changes_ =
      intervals_to_state_changes(it->out_of_service_, it->status_);

  auto tasks = hash_set<std::pair<n::location_idx_t, osr::direction>>{};
  loc_rtree_.in_radius(it->pos_, kElevatorUpdateRadius,
                       [&](n::location_idx_t const l) {
                         tasks.emplace(l, osr::direction::kForward);
                         tasks.emplace(l, osr::direction::kBackward);
                       });

  auto new_e = elevators{w_, elevator_nodes_, std::move(elevators_copy)};
  auto new_rtt = n::rt::create_rt_timetable(tt_, rtt->base_day_);
  update_rtt_td_footpaths(
      w_, l_, pl_, tt_, loc_rtree_, new_e, matches_, tasks, rtt, new_rtt,
      std::chrono::seconds{c_.timetable_.value().max_footpath_length_ * 60});

  rt_ = std::make_shared<rt>(
      std::make_unique<n::rt_timetable>(std::move(new_rtt)),
      std::make_unique<elevators>(std::move(new_e)),
      std::move(rt_->railviz_rt_));

  return json::string{{"success", true}};
}

}  // namespace motis::ep