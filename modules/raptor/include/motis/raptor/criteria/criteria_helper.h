#pragma once

#include "motis/core/access/error.h"
#include "motis/protocol/RoutingRequest_generated.h"

namespace motis::raptor {
#define MAKE_ENUM_VALUE(VAL, ACCESSOR) VAL,

#define MAKE_MC_CPU_RAPTOR_TEMPLATE_INSTANCE(VAL, ACCESSOR)          \
  template void invoke_mc_cpu_raptor<VAL>(const raptor_query& query, \
                                          raptor_statistics&);

#define CASE_SEARCH_TYPE_TO_ENUM(VAL, ACCESSOR) \
  case routing::SearchType::SearchType_##VAL:   \
    return ACCESSOR::VAL;

#define CASE_ENUM_TO_STRING(VAL, ACCESSOR) \
  case ACCESSOR::VAL:                      \
    return #VAL;

#define CASE_TRAIT_SIZE_FOR_CRITERIA_CONFIG(VAL, ACCESSOR) \
  case ACCESSOR::VAL:                                      \
    return VAL::TRAITS_SIZE;

#define CASE_CRITERIA_CONFIG_TO_CPU_INVOKE(VAL, ACCESSOR) \
  case ACCESSOR::VAL:                                     \
    return raptor_gen<VAL>(                               \
        q, stats, sched, meta_info, tt,                   \
        [&](raptor_query& q) { return invoke_mc_cpu_raptor<VAL>(q, stats); });

#define CASE_CRITERIA_CONFIG_TO_GPU_INVOKE(VAL, ACCESSOR)                    \
  case ACCESSOR::VAL:                                                        \
    return raptor_gen<VAL>(q, stats, sched, meta_info, tt, [&](d_query& q) { \
      invoke_mc_gpu_raptor<VAL>(q);                                          \
      stats = *q.mem_->active_host_->stats_;                                 \
    });

#if defined(MOTIS_CUDA)

#define INIT_HOST_AND_DEVICE_MEMORY(VAL, ACCESSOR)                         \
  host_memories_.emplace(ACCESSOR::VAL, std::make_unique<host_memory>(     \
                                            stop_count, ACCESSOR::VAL));   \
  device_memories_.emplace(ACCESSOR::VAL, std::make_unique<device_memory>( \
                                              stop_count, ACCESSOR::VAL,   \
                                              route_count, max_add_starts));

#endif

}  // namespace motis::raptor