#pragma once

#include "gtest/gtest.h"

#include "motis/core/schedule/schedule.h"
#include "motis/core/journey/journey.h"
#include "motis/loader/loader.h"
#include "motis/loader/loader_options.h"
#include "motis/raptor/cpu/mc_cpu_raptor.h"
#include "motis/raptor/get_raptor_timetable.h"
#include "motis/raptor/raptor_query.h"
#include "motis/raptor/raptor_search.h"
#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

class multi_criteria_raptor_test : public ::testing::Test {
protected:
  explicit multi_criteria_raptor_test(loader::loader_options opts)
      : opts_{std::move(opts)}, rp_meta_info_{nullptr}, rp_tt_{nullptr} {};

  void SetUp() override {
    sched_ = loader::load_schedule(opts_);
    manipulate_schedule();
    auto [meta, tt] = get_raptor_timetable(*sched_);
    rp_meta_info_ = std::move(meta);
    rp_tt_ = std::move(tt);
    check_mock_on_rp_sched();

#if defined(MOTIS_CUDA)
    h_gtt_ = get_host_gpu_timetable(*rp_tt_);
    d_gtt_ = get_device_gpu_timetable(*h_gtt_);
    mem_store_.init(*rp_meta_info_, *rp_tt_, 1);
#endif
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

  template <raptor_criteria_config CriteriaConfig>
  base_query get_base_query(std::unique_ptr<raptor_meta_info> const& meta_info,
                            time const dep, std::string const& eva_from,
                            std::string const& eva_to) {
    base_query bq{};
    bq.source_time_begin_ = dep;
    bq.source_time_end_ = dep;
    bq.source_ = meta_info->eva_to_raptor_id_.at(eva_from);
    bq.target_ = meta_info->eva_to_raptor_id_.at(eva_to);
    bq.criteria_config_ = CriteriaConfig;

    return bq;
  }

  template<raptor_criteria_config Config>
  std::vector<journey> execute_mc_cpu_raptor(
      schedule const& sched, std::unique_ptr<raptor_meta_info> const& meta_info,
      std::unique_ptr<raptor_timetable> const& tt, time const dep,
      std::string const& eva_from, std::string const& eva_to) {

    auto const bq = get_base_query<Config>(
        meta_info, dep, eva_from, eva_to);
    auto q = raptor_query{bq, *rp_meta_info_, *rp_tt_};

    raptor_statistics stats;
    auto const j = search_dispatch<implementation_type::CPU>(
        q, stats, *sched_, *rp_meta_info_, *rp_tt_);

    return j;
  }

  schedule_ptr sched_;
  loader::loader_options opts_;
  std::unique_ptr<raptor_meta_info> rp_meta_info_;
  std::unique_ptr<raptor_timetable> rp_tt_;

#if defined(MOTIS_CUDA)
  template<raptor_criteria_config Config>
  std::vector<journey> execute_mc_gpu_raptor(
      schedule const& sched, std::unique_ptr<raptor_meta_info> const& meta_info,
      std::unique_ptr<raptor_timetable> const& tt, time dep,
      std::string const& eva_from, std::string const& eva_to) {

    auto const bq = get_base_query<Config>(
        meta_info, dep, eva_from, eva_to);
    loaned_mem loan(mem_store_);

    d_query q(bq, *rp_meta_info_, loan.mem_, *d_gtt_);

    raptor_statistics stats;
    auto const j = search_dispatch<implementation_type::GPU>(
        q, stats, *sched_, *rp_meta_info_, *rp_tt_);
    return j;
  }

  std::unique_ptr<host_gpu_timetable> h_gtt_;
  std::unique_ptr<device_gpu_timetable> d_gtt_;
  memory_store mem_store_;
#endif
};

}  // namespace motis::raptor