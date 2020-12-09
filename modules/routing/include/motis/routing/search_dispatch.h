#pragma once

#include "motis/core/schedule/edges.h"

#include "motis/protocol/RoutingRequest_generated.h"
#include "motis/routing/error.h"
#include "motis/routing/search.h"
#include "motis/routing/start_label_generators/ontrip_gen.h"
#include "motis/routing/start_label_generators/pretrip_gen.h"

namespace motis::routing {

template <typename T>
struct get_search_dir {};

template <template <search_dir, size_t, bool, typename...> class Label,
          size_t MaxBucket, bool PostSearchDominanceEnabled, typename... Args>
struct get_search_dir<
    Label<search_dir::FWD, MaxBucket, PostSearchDominanceEnabled, Args...>> {
  static constexpr auto V = search_dir::FWD;
};

template <template <search_dir, size_t, bool, typename...> class Label,
          size_t MaxBucket, bool PostSearchDominanceEnabled, typename... Args>
struct get_search_dir<
    Label<search_dir::BWD, MaxBucket, PostSearchDominanceEnabled, Args...>> {
  static constexpr auto V = search_dir::BWD;
};

template <typename Label, template <search_dir, typename> class Gen>
search_result get_connections(search_query const& q) {
  return search<get_search_dir<Label>::V, Gen<get_search_dir<Label>::V, Label>,
                Label>::get_connections(q);
}

template <search_dir Dir, template <search_dir, typename> class Gen>
inline search_result search_dispatch(search_query const& q,
                                     SearchType const t) {
  switch (t) {
    case SearchType_Default:
      if (Dir == search_dir::FWD) {
        return get_connections<default_label<Dir>, Gen>(q);
      } else {
        return get_connections<default_simple_label<Dir>, Gen>(q);
      }
    case SearchType_SingleCriterion:
      return get_connections<single_criterion_label<Dir>, Gen>(q);
    case SearchType_SingleCriterionNoIntercity:
      return get_connections<single_criterion_no_intercity_label<Dir>, Gen>(q);
    case SearchType_LateConnections:
      return get_connections<late_connections_label<Dir>, Gen>(q);
    case SearchType_LateConnectionsTest:
      return get_connections<late_connections_label_for_tests<Dir>, Gen>(q);
    case SearchType_Accessibility:
      return get_connections<accessibility_label<Dir>, Gen>(q);
#ifdef MOTIS_CAPACITY_IN_SCHEDULE
    case SearchType_PerceivedTravelTime:
      return get_connections<perceived_travel_time_label<Dir>, Gen>(q);
    case SearchType_LoadSum:
      return get_connections<load_sum_label<Dir>, Gen>(q);
#endif
    default: break;
  }
  throw std::system_error(error::search_type_not_supported);
}

inline search_result search_dispatch(search_query const& q, Start s,
                                     SearchType const t, SearchDir d) {
  switch (s) {
    case Start_PretripStart:
      if (d == SearchDir_Forward) {
        return search_dispatch<search_dir::FWD, pretrip_gen>(q, t);
      } else {
        return search_dispatch<search_dir::BWD, pretrip_gen>(q, t);
      }
    case Start_OntripStationStart:
    case Start_OntripTrainStart:
      if (d == SearchDir_Forward) {
        return search_dispatch<search_dir::FWD, ontrip_gen>(q, t);
      } else {
        return search_dispatch<search_dir::BWD, ontrip_gen>(q, t);
      }
      break;
    case Start_NONE: assert(false);
  }
  return {};
}

}  // namespace motis::routing
