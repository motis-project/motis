#pragma once

#include "gtest/gtest.h"

#include "motis/loader/loader.h"
#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"
#include "motis/loader/loader_options.h"
#include "motis/raptor/cpu/mc_cpu_raptor.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_result.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/reconstructor.h"
#include "motis/raptor/get_raptor_timetable.h"

namespace motis::raptor {

class cpu_raptor_test : public ::testing::Test {
protected:
  explicit cpu_raptor_test(loader::loader_options opts)
      : opts_{std::move(opts)}, rp_meta_info_{nullptr},
        rp_tt_{nullptr}{};

  void SetUp() override {
    sched_ = loader::load_schedule(opts_);
    manipulate_schedule();
    auto [ meta, tt ] = get_raptor_timetable(*sched_);
    rp_meta_info_ = std::move(meta);
    rp_tt_ = std::move(tt);
    check_mock_on_rp_sched();
  }

  virtual void manipulate_schedule() = 0;
  virtual void check_mock_on_rp_sched() = 0;

  uint32_t get_raptor_r_id(std::string const& gtfs_trip_id) {
    auto& trip2route = rp_meta_info_->route_mapping_.trip_dbg_to_route_trips_;
    for (auto const& entry : trip2route) {
      if (entry.first.starts_with(gtfs_trip_id)) {
        if (entry.second.size() > 1) {
          throw std::runtime_error{
              "No unique mapping between route ids and gtfs ids!"};
        } else {
          return entry.second.begin()->first;
        }
      }
    }

    throw std::runtime_error{"No entries in trip2route! => GTFS id is unknown"};
  }

  schedule_ptr sched_;
  loader::loader_options opts_;
  std::unique_ptr<raptor_meta_info> rp_meta_info_;
  std::unique_ptr<raptor_timetable> rp_tt_;
};

template <typename CriteriaConfig>
std::vector<journey> execute_mc_cpu_raptor(
    schedule const& sched, std::unique_ptr<raptor_meta_info> const& meta_info,
    std::unique_ptr<raptor_timetable> const& tt, time dep,
    std::string const& eva_from, std::string const& eva_to) {

  base_query bq{};
  bq.source_time_begin_ = dep;
  bq.source_time_end_ = bq.source_time_begin_;
  bq.source_ = meta_info->eva_to_raptor_id_.at(eva_from);
  bq.target_ = meta_info->eva_to_raptor_id_.at(eva_to);

  raptor_query q{bq, *meta_info, *tt};
  q.result_.reset();
  q.result_ = std::make_unique<raptor_result>(tt->stop_count(), raptor_criteria_config::MaxOccupancy);
  raptor_statistics st;

  invoke_mc_cpu_raptor<CriteriaConfig>(q, st);
  reconstructor<CriteriaConfig> rc{sched, *meta_info, *tt};
  rc.add(q);
  return rc.get_journeys();
}

}  // namespace motis::raptor