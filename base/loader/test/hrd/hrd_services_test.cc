#include <cinttypes>
#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "utl/parser/cstr.h"

#include "motis/core/common/date_time_util.h"
#include "motis/loader/hrd/model/hrd_service.h"

#include "./paths.h"
#include "./test_spec_test.h"

using namespace utl;

namespace motis::loader::hrd {

TEST(loader_hrd_hrd_services, simple_ranges) {
  test_spec const services_file(SCHEDULES / "hand-crafted" / "fahrten",
                                "services-1.101");
  test_spec const services_file_new(SCHEDULES / "hand-crafted_new" / "fahrten",
                                    "services-1.txt");
  auto services_old = services_file.get_hrd_services(hrd_5_00_8);
  auto services_new = services_file_new.get_hrd_services(hrd_5_20_26);
  for (auto const& services : {services_old, services_new}) {
    ASSERT_TRUE(services.size() == 1);
    auto const& service = services[0];
    ASSERT_TRUE(service.sections_.size() == 5);
    std::for_each(
        std::begin(service.sections_), std::end(service.sections_),
        [](hrd_service::section const& s) {
          ASSERT_TRUE(s.traffic_days_ == std::vector<int>({2687}));
          ASSERT_TRUE(s.train_num_ == 2292);
          ASSERT_TRUE(s.admin_ == "80____");
          ASSERT_TRUE(s.attributes_ == std::vector<hrd_service::attribute>(
                                           {hrd_service::attribute(0, "BT"),
                                            hrd_service::attribute(0, "FR"),
                                            hrd_service::attribute(0, "G ")}));
          ASSERT_TRUE(s.category_ == std::vector<cstr>({"IC "}));
          EXPECT_EQ(std::vector<cstr>({"381"}), s.line_information_);
        });
    ASSERT_TRUE(service.stops_.size() == 6);

    auto stop = service.stops_[0];
    ASSERT_TRUE(stop.eva_num_ == 8000096);
    ASSERT_TRUE(stop.arr_.time_ == hrd_service::NOT_SET);
    ASSERT_TRUE(stop.dep_.time_ == 965);
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[1];
    ASSERT_TRUE(stop.eva_num_ == 8000156);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1644));
    ASSERT_TRUE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hhmm_to_min(1646));
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[2];
    ASSERT_TRUE(stop.eva_num_ == 8000377);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1659));
    ASSERT_TRUE(!stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hhmm_to_min(1700));
    ASSERT_TRUE(!stop.dep_.in_out_allowed_);

    stop = service.stops_[3];
    ASSERT_TRUE(stop.eva_num_ == 8000031);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1708));
    ASSERT_TRUE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hhmm_to_min(1709));
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[4];
    ASSERT_TRUE(stop.eva_num_ == 8000068);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1722));
    ASSERT_TRUE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hhmm_to_min(1724));
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[5];
    ASSERT_TRUE(stop.eva_num_ == 8000105);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1740));
    ASSERT_FALSE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hrd_service::NOT_SET);
  }
}

TEST(loader_hrd_hrd_services, complex_ranges) {
  test_spec const services_file(SCHEDULES / "hand-crafted" / "fahrten",
                                "services-2.101");
  test_spec const services_file_new(SCHEDULES / "hand-crafted_new" / "fahrten",
                                    "services-2.txt");

  auto services_old = services_file.get_hrd_services(hrd_5_00_8);
  auto services_new = services_file_new.get_hrd_services(hrd_5_20_26);

  for (auto const& services : {services_old, services_new}) {
    ASSERT_TRUE(services.size() == 1);

    auto const& service = services[0];
    ASSERT_TRUE(service.sections_.size() == 2);

    auto section = service.sections_[0];
    ASSERT_TRUE(section.train_num_ == 2292);
    ASSERT_TRUE(section.admin_ == "80____");
    ASSERT_TRUE(section.attributes_ == std::vector<hrd_service::attribute>(
                                           {hrd_service::attribute(0, "FR"),
                                            hrd_service::attribute(0, "G ")}));
    ASSERT_TRUE(section.category_ == std::vector<cstr>({"IC "}));
    ASSERT_TRUE(section.line_information_ == std::vector<cstr>({"381"}));
    ASSERT_TRUE(section.traffic_days_ == std::vector<int>({0}));

    section = service.sections_[1];
    ASSERT_TRUE(section.train_num_ == 2293);
    ASSERT_TRUE(section.admin_ == "81____");
    ASSERT_TRUE(section.attributes_ ==
                std::vector<hrd_service::attribute>(
                    {hrd_service::attribute(1337, "BT")}));
    ASSERT_TRUE(section.category_ == std::vector<cstr>({"IC "}));
    ASSERT_TRUE(section.line_information_ == std::vector<cstr>({"381"}));
    ASSERT_TRUE(section.traffic_days_ == std::vector<int>({2687}));

    ASSERT_TRUE(service.stops_.size() == 3);

    auto stop = service.stops_[0];
    ASSERT_TRUE(stop.eva_num_ == 8000096);
    ASSERT_TRUE(stop.arr_.time_ == hrd_service::NOT_SET);
    ASSERT_TRUE(stop.dep_.time_ == 965);
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[1];
    ASSERT_TRUE(stop.eva_num_ == 8000068);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1722));
    ASSERT_TRUE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hhmm_to_min(1724));
    ASSERT_TRUE(stop.dep_.in_out_allowed_);

    stop = service.stops_[2];
    ASSERT_TRUE(stop.eva_num_ == 8000105);
    ASSERT_TRUE(stop.arr_.time_ == hhmm_to_min(1740));
    ASSERT_TRUE(stop.arr_.in_out_allowed_);
    ASSERT_TRUE(stop.dep_.time_ == hrd_service::NOT_SET);
  }
}

TEST(loader_hrd_hrd_services, new_line_format) {
  test_spec const services_file(SCHEDULES / "hand-crafted_new" / "fahrten",
                                "services-4.txt");
  auto services = services_file.get_hrd_services(hrd_5_20_26);
  ASSERT_TRUE(services.size() == 1);
  auto const& service = services[0];
  ASSERT_TRUE(service.sections_.size() == 2);
  ASSERT_TRUE(service.sections_[0].line_information_ ==
              std::vector<cstr>({"381AIRB"}));
  ASSERT_TRUE(service.sections_[1].line_information_ ==
              std::vector<cstr>({"381AIRC"}));
}

TEST(loader_hrd_hrd_services, indices) {
  test_spec const services_file(SCHEDULES / "single-index-bus" / "fahrten",
                                "services.101");

  auto services = services_file.get_hrd_services(hrd_5_00_8);
  ASSERT_TRUE(services.size() == 1);

  auto const& service = services[0];
  ASSERT_TRUE(service.sections_.size() == 51);

  std::for_each(
      std::begin(service.sections_), std::end(service.sections_),
      [](hrd_service::section const& section) {
        ASSERT_TRUE(section.train_num_ == 0);
        ASSERT_TRUE(section.admin_ == "rmvNOL");
        ASSERT_TRUE(section.attributes_ ==
                    std::vector<hrd_service::attribute>(
                        {hrd_service::attribute(0, "OB")}));
        ASSERT_TRUE(section.category_ == std::vector<cstr>({"Bus"}));
        ASSERT_TRUE(section.line_information_ == std::vector<cstr>({"84/85"}));
        ASSERT_TRUE(section.traffic_days_ == std::vector<int>({2310}));
      });
}

TEST(loader_hrd_hrd_services, time_prefixes) {
  test_spec const services_file(SCHEDULES / "complex-ranges" / "fahrten",
                                "services.101");

  auto services = services_file.get_hrd_services(hrd_5_00_8);
  ASSERT_TRUE(services.size() == 1);

  auto const& service = services[0];
  ASSERT_TRUE(service.sections_.size() == 16);

  std::for_each(
      std::begin(service.sections_), std::end(service.sections_),
      [](hrd_service::section const& section) {
        ASSERT_TRUE(section.train_num_ == 0);
        ASSERT_TRUE(section.admin_ == "rmv106");
        ASSERT_TRUE(section.category_ == std::vector<cstr>({"rfb"}));
        ASSERT_TRUE(section.line_information_ == std::vector<cstr>({"56"}));
        ASSERT_TRUE(section.traffic_days_ == std::vector<int>({32283}));
        ASSERT_TRUE(section.attributes_ == std::vector<hrd_service::attribute>(
                                               {{0, "g5"}, {0, "j4"}}));
      });
}

}  // namespace motis::loader::hrd
