#pragma once

#include "motis/module/module.h"

#include "motis/csa/csa_implementation_type.h"

#ifdef MOTIS_CUDA
#include "motis/csa/gpu/gpu_timetable.h"
#endif

namespace motis::csa {

struct csa_timetable;

struct csa : public motis::module::module {
  csa();
  ~csa() override;

  csa(csa const&) = delete;
  csa& operator=(csa const&) = delete;

  csa(csa&&) = delete;
  csa& operator=(csa&&) = delete;

  void init(motis::module::registry&) override;

  csa_timetable const* get_timetable() const;

  motis::module::msg_ptr route(motis::module::msg_ptr const&,
                               implementation_type,
                               bool use_profile_search = false) const;

#ifdef MOTIS_CUDA
  bool bridge_zero_duration_connections_{true};
  bool add_footpath_connections_{true};
#else
  bool bridge_zero_duration_connections_{false};
  bool add_footpath_connections_{false};
#endif
  std::unique_ptr<csa_timetable> timetable_;
};

}  // namespace motis::csa
