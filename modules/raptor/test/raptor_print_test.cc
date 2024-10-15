#include "gtest/gtest.h"

#include "motis/test/schedule/simple_realtime.h"

#include "motis/loader/loader.h"
#include "motis/raptor/get_raptor_timetable.h"
#include "motis/raptor/print_raptor.h"
#include "motis/raptor/raptor_result.h"

using namespace motis;
using namespace motis::test;
using namespace motis::raptor;
using motis::test::schedule::simple_realtime::dataset_opt;

TEST(print_raptor_itest, just_print) {
  auto const sched = loader::load_schedule(dataset_opt);
  auto const [meta_info, raptor_timetable] = get_raptor_timetable(*sched);

  auto const station_string = get_string(0, *meta_info);

  print_station(0, *meta_info);
  print_route(0, *raptor_timetable);
  print_route_format(0, *raptor_timetable);
  print_routes(std::vector<route_id>{0, 1}, *raptor_timetable);
  print_footpaths(*raptor_timetable);
  print_routes_format(std::vector<route_id>{0, 1}, *raptor_timetable);
  routes_from_station(0, *raptor_timetable);
  get_routes_containing(std::vector<stop_id>{0, 1}, *raptor_timetable);
  get_routes_containing_evas(std::vector<std::string>{"8000096", "8000080"},
                             *meta_info, *raptor_timetable);
}

TEST(print_raptor_itest_results, just_print) {
  auto const sched = loader::load_schedule(dataset_opt);
  auto const [meta_info, raptor_timetable] = get_raptor_timetable(*sched);

  raptor_result result(raptor_timetable->stop_count(),
                       raptor_criteria_config::Default);
  print_station_arrivals(0, result);
  print_route_arrivals(0, *raptor_timetable, result[0]);
  ASSERT_TRUE(is_reset(result));
}
