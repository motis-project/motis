#pragma once

#include "geo/latlng.h"

#include "motis/core/schedule/time.h"
#include "motis/core/journey/journey.h"

#include "motis/module/context/motis_call.h"
#include "motis/module/message.h"

#include "motis/intermodal/mumo_edge.h"
#include "motis/intermodal/ppr_profiles.h"
#include "motis/intermodal/query_bounds.h"

namespace motis::intermodal {

struct direct_connection {
  direct_connection() = default;
  direct_connection(mumo_type type, unsigned duration, unsigned accessibility,
                    int mumo_id = -1)
      : type_{type},
        duration_{duration},
        accessibility_{accessibility},
        mumo_id_{mumo_id} {}

  mumo_type type_{mumo_type::FOOT};
  unsigned duration_{0U};  // minutes
  unsigned accessibility_{0U};
  int mumo_id_{0};
};

std::vector<direct_connection> get_direct_connections(
    query_start const& q_start, query_dest const& q_dest,
    IntermodalRoutingRequest const* req, ppr_profiles const& profiles,
    std::vector<mumo_edge const*> const& edge_mapping);

std::size_t remove_dominated_journeys(
    std::vector<journey>& journeys,
    std::vector<direct_connection> const& direct);

void add_direct_connections(std::vector<journey>& journeys,
                            std::vector<direct_connection> const& direct,
                            query_start const& q_start,
                            query_dest const& q_dest,
                            IntermodalRoutingRequest const* req);

flatbuffers::Offset<DirectConnection> to_fbs(
    flatbuffers::FlatBufferBuilder& fbb, direct_connection const& c);

}  // namespace motis::intermodal
