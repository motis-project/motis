#pragma once

#include <memory>
#include <string>
#include <vector>

#include "motis/module/module.h"

namespace motis::nigiri {

struct nigiri : public motis::module::module {
  nigiri();
  ~nigiri() override;

  nigiri(nigiri const&) = delete;
  nigiri& operator=(nigiri const&) = delete;

  nigiri(nigiri&&) = delete;
  nigiri& operator=(nigiri&&) = delete;

  void init(motis::module::registry&) override;
  void import(motis::module::import_dispatcher&) override;
  bool import_successful() const override { return import_successful_; }

private:
  void register_gtfsrt_timer(motis::module::dispatcher&);
  void update_gtfsrt();

  bool import_successful_{false};

  struct impl;
  std::unique_ptr<impl> impl_;
  bool no_cache_{false};
  bool adjust_footpaths_{true};
  bool merge_duplicates_{false};
  std::string first_day_;
  std::string default_timezone_;
  std::uint16_t num_days_{2U};
  bool lookup_{true};
  bool guesser_{true};
  bool railviz_{true};
  bool routing_{true};
  unsigned link_stop_distance_{100U};
  bool use_stationfilter_{false};
  bool time_consistency_{false};
  bool percentage_filter_{false};
  double percent_for_filter_{0.1};
  bool weighted_filter_{false};
  bool line_filter_{false};
  std::vector<std::string> gtfsrt_urls_;
  std::vector<std::string> gtfsrt_paths_;
  unsigned gtfsrt_update_interval_sec_{60U};
  bool gtfsrt_incremental_{false};
};

}  // namespace motis::nigiri
