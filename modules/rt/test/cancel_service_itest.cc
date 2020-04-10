#include "gtest/gtest.h"

#include "motis/core/access/time_access.h"
#include "motis/core/access/trip_access.h"
#include "motis/ris/risml/risml_parser.h"
#include "motis/rt/separate_trip.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/invalid_realtime.h"

using namespace motis;
using namespace motis::rt;
using namespace motis::test;
using namespace motis::module;
using motis::test::schedule::invalid_realtime::dataset_opt_no_rules;

struct rt_cancel_service_test : public motis_instance_test {
  rt_cancel_service_test()
      : motis::test::motis_instance_test(
            dataset_opt_no_rules, {"ris", "rt"},
            {"--ris.input=test/schedule/invalid_realtime/risml/cancel.xml",
             "--ris.init_time=2015-11-24T11:00:00"}) {}
};

TEST_F(rt_cancel_service_test, simple) {
  auto trp = get_trip(sched(), "0000001", 1, unix_time(1010), "0000005",
                      unix_time(1400), "381");

  auto const valid = std::map<std::string, bool>{{"0000001", true},
                                                 {"0000002", true},
                                                 {"0000003", false},
                                                 {"0000004", false},
                                                 {"0000005", false}};
  for (auto const& trp_e : *trp->edges_) {
    edge const* e = trp_e.get_edge();

    auto const dep = ev_key{e, trp->lcon_idx_, event_type::DEP};
    auto const& id = sched().stations_[dep.get_station_idx()]->eva_nr_;
    EXPECT_EQ(valid.at(id.str()), dep.lcon()->valid_);
  }
}
