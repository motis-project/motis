#include "motis/paxforecast/messages.h"

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cista/reflection/comparable.h"

#include "utl/enumerate.h"
#include "utl/to_vec.h"

#include "motis/hash_map.h"

#include "motis/core/access/service_access.h"
#include "motis/core/access/trip_iterator.h"
#include "motis/core/conv/station_conv.h"
#include "motis/core/conv/trip_conv.h"

#include "motis/paxmon/messages.h"

using namespace motis::module;
using namespace flatbuffers;
using namespace motis::paxmon;

namespace motis::paxforecast {

Offset<PassengerGroupForecast> get_passenger_group_forecast(
    FlatBufferBuilder& fbb, schedule const& sched, passenger_group const& grp,
    group_simulation_result const& group_result) {
  return CreatePassengerGroupForecast(
      fbb, to_fbs(sched, fbb, grp),
      fbs_localization_type(*group_result.localization_),
      to_fbs(sched, fbb, *group_result.localization_),
      fbb.CreateVector(
          utl::to_vec(group_result.alternatives_, [&](auto const& alt) {
            return CreateForecastAlternative(
                fbb, to_fbs(sched, fbb, alt.first->compact_journey_),
                alt.second);
          })));
}

Offset<Vector<CdfEntry const*>> cdf_to_fbs(FlatBufferBuilder& fbb,
                                           pax_cdf const& cdf) {
  auto entries = std::vector<CdfEntry>{};
  entries.reserve(cdf.data_.size());
  auto last_prob = 0.0F;
  for (auto const& [pax, prob] : utl::enumerate(cdf.data_)) {
    if (prob != last_prob) {
      entries.emplace_back(pax, prob);
      last_prob = prob;
    }
  }
  return fbb.CreateVectorOfStructs(entries);
}

CapacityType get_capacity_type(motis::paxmon::edge const* e) {
  if (e->has_unknown_capacity()) {
    return CapacityType_Unknown;
  } else if (e->has_unlimited_capacity()) {
    return CapacityType_Unlimited;
  } else {
    return CapacityType_Known;
  }
}

struct service_info {
  CISTA_COMPARABLE();

  std::string name_;
  std::string_view category_;
  std::uint32_t train_nr_{};
  std::string_view line_;
  std::string_view provider_;
  service_class clasz_{service_class::OTHER};
};

Offset<ServiceInfo> to_fbs(FlatBufferBuilder& fbb, service_info const& si) {
  return CreateServiceInfo(
      fbb, fbb.CreateString(si.name_), fbb.CreateString(si.category_),
      si.train_nr_, fbb.CreateString(si.line_), fbb.CreateString(si.provider_),
      static_cast<service_class_t>(si.clasz_));
}

std::vector<std::pair<service_info, unsigned>> get_service_infos(
    schedule const& sched, trip_forecast const& tfc) {
  mcd::hash_map<service_info, unsigned> si_counts;
  for (auto const& section : motis::access::sections(tfc.trp_)) {
    auto const& fc = section.fcon();
    for (auto ci = fc.con_info_; ci != nullptr; ci = ci->merged_with_) {
      auto const si = service_info{
          get_service_name(sched, ci),
          sched.categories_.at(ci->family_)->name_.view(),
          output_train_nr(ci->train_nr_, ci->original_train_nr_),
          ci->line_identifier_.view(),
          ci->provider_ != nullptr ? ci->provider_->full_name_.view()
                                   : std::string_view{},
          fc.clasz_};
      ++si_counts[si];
    }
  }
  auto sis = utl::to_vec(si_counts, [](auto const& e) {
    return std::make_pair(e.first, e.second);
  });
  std::sort(begin(sis), end(sis),
            [](auto const& a, auto const& b) { return a.second > b.second; });
  return sis;
}

Offset<EdgeForecast> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, edge_forecast const& efc) {
  auto const from = efc.edge_->from(g);
  auto const to = efc.edge_->to(g);
  return CreateEdgeForecast(fbb, to_fbs(fbb, from->get_station(sched)),
                            to_fbs(fbb, to->get_station(sched)),
                            motis_to_unixtime(sched, from->schedule_time()),
                            motis_to_unixtime(sched, from->current_time()),
                            motis_to_unixtime(sched, to->schedule_time()),
                            motis_to_unixtime(sched, to->current_time()),
                            get_capacity_type(efc.edge_), efc.edge_->capacity(),
                            cdf_to_fbs(fbb, efc.forecast_cdf_), efc.updated_,
                            efc.possibly_over_capacity_,
                            efc.expected_passengers_);
}

Offset<TripForecast> to_fbs(FlatBufferBuilder& fbb, schedule const& sched,
                            graph const& g, trip_forecast const& tfc) {
  return CreateTripForecast(
      fbb, to_fbs(sched, fbb, tfc.trp_),
      to_fbs(fbb, *sched.stations_.at(tfc.trp_->id_.primary_.get_station_id())),
      to_fbs(fbb,
             *sched.stations_.at(tfc.trp_->id_.secondary_.target_station_id_)),
      fbb.CreateVector(
          utl::to_vec(get_service_infos(sched, tfc),
                      [&](auto const& sip) { return to_fbs(fbb, sip.first); })),
      fbb.CreateVector(utl::to_vec(tfc.edges_, [&](auto const& efc) {
        return to_fbs(fbb, sched, g, efc);
      })));
}

Offset<Vector<Offset<TripForecast>>> to_fbs(FlatBufferBuilder& fbb,
                                            schedule const& sched,
                                            graph const& g,
                                            load_forecast const& lfc) {
  return fbb.CreateVector(utl::to_vec(
      lfc.trips_, [&](auto const& tfc) { return to_fbs(fbb, sched, g, tfc); }));
}

msg_ptr make_passenger_forecast_msg(schedule const& sched,
                                    motis::paxmon::paxmon_data const& data,
                                    simulation_result const& sim_result,
                                    load_forecast const& lfc) {
  message_creator fbb;
  fbb.create_and_finish(
      MsgContent_PassengerForecast,
      CreatePassengerForecast(fbb, sched.system_time_,
                              fbb.CreateVector(utl::to_vec(
                                  sim_result.group_results_,
                                  [&](auto const& entry) {
                                    return get_passenger_group_forecast(
                                        fbb, sched, *entry.first, entry.second);
                                  })),
                              to_fbs(fbb, sched, data.graph_, lfc))
          .Union(),
      "/paxforecast/passenger_forecast");
  return make_msg(fbb);
}

}  // namespace motis::paxforecast
