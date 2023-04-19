#include <cinttypes>
#include <iostream>

#include "boost/filesystem.hpp"
#include "boost/range/iterator_range.hpp"

#include "gtest/gtest.h"

#include "flatbuffers/flatbuffers.h"

#include "utl/erase_if.h"

#include "motis/core/common/logging.h"

#include "motis/loader/hrd/builder/bitfield_builder.h"
#include "motis/loader/hrd/builder/rule_service_builder.h"
#include "motis/loader/hrd/model/split_service.h"
#include "motis/loader/hrd/parser/bitfields_parser.h"
#include "motis/loader/hrd/parser/merge_split_rules_parser.h"
#include "motis/loader/hrd/parser/service_parser.h"
#include "motis/loader/hrd/parser/through_services_parser.h"
#include "motis/loader/util.h"
#include "motis/schedule-format/RuleService_generated.h"

#include "./paths.h"

namespace motis::loader::hrd {

using namespace boost::filesystem;
using namespace motis::logging;

class rule_services_test : public ::testing::Test {
protected:
  explicit rule_services_test(std::string schedule_name)
      : schedule_name_(std::move(schedule_name)) {}

  void SetUp() override {
    path const root = SCHEDULES / schedule_name_;
    path const stamm = root / "stamm";
    path const fahrten = root / "fahrten";

    // load bitfields
    flatbuffers64::FlatBufferBuilder const fbb;
    data_.emplace_back(stamm / "bitfield.101");
    bitfield_builder const bb(parse_bitfields(data_.back(), hrd_5_00_8));

    // load service rules
    service_rules rs;
    data_.emplace_back(stamm / "durchbi.101");
    parse_through_service_rules(data_.back(), bb.hrd_bitfields_, rs,
                                hrd_5_00_8);
    data_.emplace_back(stamm / "vereinig_vt.101");
    parse_merge_split_service_rules(data_.back(), bb.hrd_bitfields_, rs,
                                    hrd_5_00_8);

    // load services and create rule services
    rsb_ = rule_service_builder(rs);
    std::vector<path> services_files;
    collect_files(fahrten, "", services_files);
    for (auto const& services_file : services_files) {
      data_.emplace_back(services_file);
      for_each_service(
          data_.back(), bb.hrd_bitfields_,
          [&](hrd_service const& s) { rsb_.add_service(s); },
          [](std::size_t) {}, hrd_5_00_8);
    }
    rsb_.resolve_rule_services();

    // remove all remaining services that does not have any traffic day left
    utl::erase_if(rsb_.origin_services_,
                  [](std::unique_ptr<hrd_service> const& service_ptr) {
                    return service_ptr->traffic_days_.none();
                  });
  }

  std::string schedule_name_;
  rule_service_builder rsb_;

private:
  std::vector<loaded_file> data_;
};

class loader_ts_once : public rule_services_test {
public:
  loader_ts_once() : rule_services_test("ts-once") {}
};

class loader_ts_twice : public rule_services_test {
public:
  loader_ts_twice() : rule_services_test("ts-twice") {}
};

class loader_ts_twice_all_combinations : public rule_services_test {
public:
  loader_ts_twice_all_combinations()
      : rule_services_test("ts-twice-all-combinations") {}
};

class loader_ts_2_to_1 : public rule_services_test {
public:
  loader_ts_2_to_1() : rule_services_test("ts-2-to-1") {}
};

class loader_ts_2_to_1_cycle : public rule_services_test {
public:
  loader_ts_2_to_1_cycle() : rule_services_test("ts-2-to-1-cycle") {}
};

class loader_ts_twice_2_to_1_cycle : public rule_services_test {
public:
  loader_ts_twice_2_to_1_cycle()
      : rule_services_test("ts-twice-2-to-1-cycle") {}
};

class loader_ts_passing_service : public rule_services_test {
public:
  loader_ts_passing_service() : rule_services_test("ts-passing-service") {}
};

class loader_mss_once : public rule_services_test {
public:
  loader_mss_once() : rule_services_test("mss-once") {}
};

class loader_mss_twice : public rule_services_test {
public:
  loader_mss_twice() : rule_services_test("mss-twice") {}
};

class loader_mss_many : public rule_services_test {
public:
  loader_mss_many() : rule_services_test("mss-many") {}
};

TEST_F(loader_ts_once, rule_services) {
  // check remaining services
  ASSERT_EQ(1, rsb_.origin_services_.size());

  auto const& remaining_service = rsb_.origin_services_[0].get();
  ASSERT_EQ(bitfield{"0001110"}, remaining_service->traffic_days_);

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service = rsb_.rule_services_[0];
  for (auto const& sr : rule_service.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0010001"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0010001"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_twice, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(2, rsb_.rule_services_.size());

  auto const& rule_service1 = rsb_.rule_services_[0];
  ASSERT_EQ(3, rule_service1.services_.size());
  ASSERT_EQ(2, rule_service1.rules_.size());
  for (auto const& sr : rule_service1.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s2_->traffic_days_);
  }
  auto const& rule_service2 = rsb_.rule_services_[1];
  ASSERT_EQ(2, rule_service2.services_.size());
  ASSERT_EQ(1, rule_service2.rules_.size());
  for (auto const& sr : rule_service2.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1100000"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1100000"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_twice_all_combinations, rule_services) {
  // check remaining services
  ASSERT_EQ(3, rsb_.origin_services_.size());
  auto const& service1 = rsb_.origin_services_[0];
  ASSERT_EQ(bitfield{"0000001"}, service1->traffic_days_);
  auto const& service2 = rsb_.origin_services_[1];
  ASSERT_EQ(bitfield{"0000010"}, service2->traffic_days_);
  auto const& service3 = rsb_.origin_services_[2];
  ASSERT_EQ(bitfield{"0000100"}, service3->traffic_days_);

  // check rule services
  ASSERT_EQ(3, rsb_.rule_services_.size());

  auto const& rule_service1 = rsb_.rule_services_[0];
  ASSERT_EQ(3, rule_service1.services_.size());
  ASSERT_EQ(2, rule_service1.rules_.size());
  for (auto const& sr : rule_service1.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0100000"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0100000"}, sr.s2_->traffic_days_);
  }

  auto const& rule_service2 = rsb_.rule_services_[1];
  ASSERT_EQ(2, rule_service2.services_.size());
  ASSERT_EQ(1, rule_service2.rules_.size());
  for (auto const& sr : rule_service2.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0001000"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0001000"}, sr.s2_->traffic_days_);
  }

  auto const& rule_service3 = rsb_.rule_services_[2];
  ASSERT_EQ(2, rule_service3.services_.size());
  ASSERT_EQ(1, rule_service3.rules_.size());
  for (auto const& sr : rule_service3.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0010000"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0010000"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_2_to_1, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(2, rsb_.rule_services_.size());

  auto const& rule_service1 = rsb_.rule_services_[0];
  ASSERT_EQ(2, rule_service1.services_.size());
  ASSERT_EQ(1, rule_service1.rules_.size());
  for (auto const& sr : rule_service1.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s2_->traffic_days_);
  }

  auto const& rule_service2 = rsb_.rule_services_[1];
  ASSERT_EQ(2, rule_service2.services_.size());
  ASSERT_EQ(1, rule_service2.rules_.size());
  for (auto const& sr : rule_service2.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1100000"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1100000"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_2_to_1_cycle, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service1 = rsb_.rule_services_[0];
  ASSERT_EQ(4, rule_service1.services_.size());
  ASSERT_EQ(3, rule_service1.rules_.size());
  for (auto const& sr : rule_service1.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_twice_2_to_1_cycle, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service1 = rsb_.rule_services_[0];
  ASSERT_EQ(5, rule_service1.services_.size());
  ASSERT_EQ(4, rule_service1.rules_.size());
  for (auto const& sr : rule_service1.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_ts_passing_service, rule_services) {
  // check remaining services
  ASSERT_EQ(1, rsb_.origin_services_.size());

  auto const& remaining_service = rsb_.origin_services_[0];
  ASSERT_EQ(bitfield{"1100000"}, remaining_service->traffic_days_);

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service = rsb_.rule_services_[0];
  ASSERT_EQ(2, rule_service.services_.size());
  ASSERT_EQ(1, rule_service.rules_.size());
  for (auto const& sr : rule_service.rules_) {
    ASSERT_EQ(RuleType_THROUGH, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0011111"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_mss_once, rule_services) {
  // check remaining services
  ASSERT_EQ(1, rsb_.origin_services_.size());

  auto const& remaining_service = rsb_.origin_services_[0].get();
  ASSERT_EQ(bitfield{"1111011"}, remaining_service->traffic_days_);

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service = rsb_.rule_services_[0];
  for (auto const& sr : rule_service.rules_) {
    ASSERT_EQ(RuleType_MERGE_SPLIT, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"0000100"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"0000100"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_mss_twice, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service = rsb_.rule_services_[0];
  ASSERT_EQ(3, rule_service.services_.size());
  ASSERT_EQ(2, rule_service.rules_.size());
  for (auto const& sr : rule_service.rules_) {
    ASSERT_EQ(RuleType_MERGE_SPLIT, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s2_->traffic_days_);
  }
}

TEST_F(loader_mss_many, rule_services) {
  // check remaining services
  ASSERT_EQ(0, rsb_.origin_services_.size());

  // check rule services
  ASSERT_EQ(1, rsb_.rule_services_.size());

  auto const& rule_service = rsb_.rule_services_[0];
  ASSERT_EQ(3, rule_service.services_.size());
  ASSERT_EQ(2, rule_service.rules_.size());
  for (auto const& sr : rule_service.rules_) {
    ASSERT_EQ(RuleType_MERGE_SPLIT, sr.rule_info_.type_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s1_->traffic_days_);
    ASSERT_EQ(bitfield{"1111111"}, sr.s2_->traffic_days_);
  }
}

}  // namespace motis::loader::hrd
