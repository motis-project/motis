#include "gtest/gtest.h"

#include <string_view>

#include "motis/elevators/parse_siri_fm.h"

using namespace motis;

constexpr auto kSiriFm = R"(
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Siri xmlns="http://www.siri.org.uk/siri" xmlns:ns2="http://www.ifopt.org.uk/acsb"
    xmlns:ns3="http://www.ifopt.org.uk/ifopt" xmlns:ns4="http://datex2.eu/schema/2_0RC1/2_0"
    xmlns:ns5="http://www.opengis.net/gml/3.2" version="siri:2.2">
    <ServiceDelivery>
        <ResponseTimestamp>2026-02-21T22:06:02Z</ResponseTimestamp>
        <ProducerRef>dbinfrago</ProducerRef>
        <FacilityMonitoringDelivery version="epiprt:2.1">
            <ResponseTimestamp>2026-02-21T22:06:02Z</ResponseTimestamp>
            <FacilityCondition>
                <FacilityRef>diid:02aeed9c-f8a3-1fd0-bceb-153a94e98000</FacilityRef>
                <FacilityStatus>
                    <Status>unknown</Status>
                    <Description xml:lang="en">not monitored</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-d36ed2aeb7ae</FacilityRef>
                <FacilityStatus>
                    <Status>available</Status>
                    <Description xml:lang="en">available</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-a458663cb7ac</FacilityRef>
                <FacilityStatus>
                    <Status>available</Status>
                    <Description xml:lang="en">available</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-9efb1b6717ac</FacilityRef>
                <FacilityStatus>
                    <Status>available</Status>
                    <Description xml:lang="en">available</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-9d577a00b7ac</FacilityRef>
                <FacilityStatus>
                    <Status>available</Status>
                    <Description xml:lang="en">available</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-b4967b7cf7ae</FacilityRef>
                <FacilityStatus>
                    <Status>available</Status>
                    <Description xml:lang="en">available</Description>
                </FacilityStatus>
            </FacilityCondition>
            <FacilityCondition>
                <FacilityRef>diid:02b2be0f-c1da-1eef-a490-db549df337ae</FacilityRef>
                <FacilityStatus>
                    <Status>notAvailable</Status>
                    <Description xml:lang="en">not available</Description>
                </FacilityStatus>
            </FacilityCondition>
      </FacilityMonitoringDelivery>
  </ServiceDelivery>
</Siri>
)";

TEST(motis, parse_siri_fm) {
  auto const elevators = parse_siri_fm(std::string_view{kSiriFm});
  ASSERT_EQ(7, elevators.size());

  ASSERT_TRUE(elevators[elevator_idx_t{0}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{1}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{2}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{3}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{4}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{5}].id_str_.has_value());
  ASSERT_TRUE(elevators[elevator_idx_t{6}].id_str_.has_value());
  EXPECT_EQ("diid:02aeed9c-f8a3-1fd0-bceb-153a94e98000",
            *elevators[elevator_idx_t{0}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-d36ed2aeb7ae",
            *elevators[elevator_idx_t{1}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-a458663cb7ac",
            *elevators[elevator_idx_t{2}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-9efb1b6717ac",
            *elevators[elevator_idx_t{3}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-9d577a00b7ac",
            *elevators[elevator_idx_t{4}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-b4967b7cf7ae",
            *elevators[elevator_idx_t{5}].id_str_);
  EXPECT_EQ("diid:02b2be0f-c1da-1eef-a490-db549df337ae",
            *elevators[elevator_idx_t{6}].id_str_);

  EXPECT_FALSE(elevators[elevator_idx_t{0}].status_);
  EXPECT_TRUE(elevators[elevator_idx_t{1}].status_);
  EXPECT_TRUE(elevators[elevator_idx_t{2}].status_);
  EXPECT_TRUE(elevators[elevator_idx_t{3}].status_);
  EXPECT_TRUE(elevators[elevator_idx_t{4}].status_);
  EXPECT_TRUE(elevators[elevator_idx_t{5}].status_);
  EXPECT_FALSE(elevators[elevator_idx_t{6}].status_);

  for (auto const& e : elevators) {
    EXPECT_TRUE(e.id_str_.has_value());
    EXPECT_TRUE(e.out_of_service_.empty());
    EXPECT_EQ(1, e.state_changes_.size());
    EXPECT_EQ(e.status_, e.state_changes_.front().state_);
  }
}
