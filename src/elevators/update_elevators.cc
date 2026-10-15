#include "motis/elevators/update_elevators.h"

#include "utl/verify.h"

#include "nigiri/logging.h"

#include "motis/config.h"
#include "motis/constants.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/elevators/parse_siri_fm.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;

namespace motis {

using elevator_map_t = hash_map<std::int64_t, elevator_idx_t>;

elevator_map_t to_map(vector_map<elevator_idx_t, elevator> const& elevators) {
  auto m = elevator_map_t{};
  for (auto const [i, e] : utl::enumerate(elevators)) {
    m.emplace(e.id_, elevator_idx_t{i});
  }
  return m;
}

ptr<elevators> update_elevators(config const& c,
                                data const& d,
                                std::string_view body,
                                n::rt_timetable& new_rtt) {
  auto new_e = std::make_unique<elevators>(
      *d.w_, d.elevator_osm_mapping_.get(), *d.elevator_nodes_,
      body.contains("<Siri") ? parse_siri_fm(body) : parse_fasta(body));
  auto const& old_e = *d.rt_->e_;
  auto const old_map = to_map(old_e.elevators_);
  auto const new_map = to_map(new_e->elevators_);

  auto tasks = hash_set<n::location_idx_t>{};
  auto const add_tasks = [&](std::optional<geo::latlng> const& pos) {
    if (!pos.has_value()) {
      return;
    }
    d.location_rtree_->in_radius(
        *pos, kElevatorUpdateRadius,
        [&](n::location_idx_t const l) { tasks.emplace(l); });
  };

  // Not listed => always working (default).
  auto const always_working = [](elevator const& x) {
    return x.status_ && x.out_of_service_.empty();
  };

  for (auto const& [id, e_idx] : old_map) {
    auto const it = new_map.find(id);
    if (it == end(new_map)) {
      // Elevator got removed. Update if it was not always working before.
      if (!always_working(old_e.elevators_[e_idx])) {
        add_tasks(old_e.elevators_[e_idx].pos_);
      }
    } else {
      // Elevator remained. Update if status or maintenance windows changed.
      if (new_e->elevators_[it->second].get_state_changes() !=
          old_e.elevators_[e_idx].get_state_changes()) {
        add_tasks(new_e->elevators_[it->second].pos_);
      }
    }
  }

  for (auto const& [id, e_idx] : new_map) {
    auto const it = old_map.find(id);
    if (it == end(old_map) && !always_working(new_e->elevators_[e_idx])) {
      // New elevator not seen before, not always working. Update.
      add_tasks(new_e->elevators_[e_idx].pos_);
    }
  }

  n::log(n::log_lvl::info, "motis.rt.elevators",
         "elevator update: {} routing tasks", tasks.size());

  update_rtt_td_footpaths(
      *d.w_, *d.l_, *d.pl_, *d.tt_, *d.location_rtree_, *new_e, *d.matches_,
      tasks, d.rt_->rtt_.get(), new_rtt,
      std::chrono::seconds{c.timetable_.value().max_footpath_length_ * 60});

  return new_e;
}

}  // namespace motis
