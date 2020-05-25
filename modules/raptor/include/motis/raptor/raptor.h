#pragma once

#include "motis/module/module.h"

#include "motis/core/journey/journey.h"

#include "motis/raptor/raptor_statistics.h"
#include "motis/raptor/raptor_timetable.h"

namespace motis::raptor {

struct raptor : public motis::module::module {
  raptor();
  ~raptor() override;

  raptor(raptor const&) = delete;
  raptor& operator=(raptor const&) = delete;

  raptor(raptor&&) = delete;
  raptor& operator=(raptor&&) = delete;

  void init(motis::module::registry&) override;

private:
  template <class Query>
  Query get_query(motis::routing::RoutingRequest const*, schedule const&);

  template <typename Query, typename RaptorFun>
  motis::module::msg_ptr route_generic(motis::module::msg_ptr const&,
                                       RaptorFun const&);

  motis::module::msg_ptr make_response(std::vector<journey> const&,
                                       motis::routing::RoutingRequest const*,
                                       raptor_statistics const&);

  std::unique_ptr<raptor_schedule> raptor_sched_;
  std::unique_ptr<raptor_timetable> timetable_;
  std::unique_ptr<raptor_timetable> backward_timetable_;
};

}  // namespace motis::raptor