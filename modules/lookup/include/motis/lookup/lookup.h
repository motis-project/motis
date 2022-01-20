#pragma once

#include <memory>

#include "geo/point_rtree.h"

#include "motis/module/module.h"

namespace motis::lookup {

struct lookup final : public motis::module::module {
  lookup();
  ~lookup() override;

  lookup(lookup const&) = delete;
  lookup& operator=(lookup const&) = delete;

  lookup(lookup&&) = delete;
  lookup& operator=(lookup&&) = delete;

  void init(motis::module::registry&) override;

private:
  motis::module::msg_ptr lookup_station_id(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr lookup_station(motis::module::msg_ptr const&) const;
  motis::module::msg_ptr lookup_stations(motis::module::msg_ptr const&) const;

  motis::module::msg_ptr lookup_station_events(motis::module::msg_ptr const&);
  motis::module::msg_ptr lookup_id_train(motis::module::msg_ptr const&);
  motis::module::msg_ptr lookup_meta_station(motis::module::msg_ptr const&);
  motis::module::msg_ptr lookup_meta_stations(motis::module::msg_ptr const&);
  motis::module::msg_ptr lookup_schedule_info();

  motis::module::msg_ptr lookup_ribasis(motis::module::msg_ptr const&);

  std::unique_ptr<geo::point_rtree> station_geo_index_;
};

}  // namespace motis::lookup
