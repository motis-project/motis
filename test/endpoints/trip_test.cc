#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/endpoints/trip.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

#include "../test_case.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

TEST(motis, trip_stop_naming) {
  auto [d, _] = get_test_case<test_case::generated_trip_stop_naming>();

  auto const trip_ep = utl::init_from<ep::trip>(d).value();

  auto const res = trip_ep("?tripId=20190501_10%3A00_test_T1");
  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_[0];
  EXPECT_GT(leg.legGeometry_.length_, 0);
  EXPECT_EQ("Child1A", leg.from_.name_);
  ASSERT_TRUE(leg.intermediateStops_.has_value());
  ASSERT_EQ(1, leg.intermediateStops_->size());
  EXPECT_EQ("Child1B", leg.intermediateStops_->at(0).name_);
  EXPECT_EQ("Parent2", leg.to_.name_);
  EXPECT_EQ("Parent2 Express", leg.headsign_);
  EXPECT_EQ("EN_SHORT_NAME", leg.routeShortName_);
  EXPECT_EQ("EN-R1", leg.routeLongName_);

  auto const compact_res =
      trip_ep("?tripId=20190501_10%3A00_test_T1&detailedLegs=false");
  ASSERT_EQ(1, compact_res.legs_.size());
  EXPECT_EQ("", compact_res.legs_[0].legGeometry_.points_);
  EXPECT_EQ(0, compact_res.legs_[0].legGeometry_.length_);

  auto const res_de = trip_ep("?tripId=20190501_10%3A00_test_T1&language=de");
  ASSERT_EQ(1, res_de.legs_.size());
  auto const& leg_de = res_de.legs_[0];
  EXPECT_EQ("Kind 1A", leg_de.from_.name_);
  ASSERT_TRUE(leg_de.intermediateStops_.has_value());
  ASSERT_EQ(1, leg_de.intermediateStops_->size());
  EXPECT_EQ("Kind 1B", leg_de.intermediateStops_->at(0).name_);
  EXPECT_EQ("Eltern 2", leg_de.to_.name_);
  EXPECT_EQ("Richtung Eltern Zwei", leg_de.headsign_);
  EXPECT_EQ("DE_SHORT_NAME", leg_de.routeShortName_);
  EXPECT_EQ("DE-R1", leg_de.routeLongName_);

  auto const res_fr = trip_ep("?tripId=20190501_10%3A00_test_T1&language=fr");
  ASSERT_EQ(1, res_fr.legs_.size());
  auto const& leg_fr = res_fr.legs_[0];
  EXPECT_EQ("Enfant 1A", leg_fr.from_.name_);
  ASSERT_TRUE(leg_fr.intermediateStops_.has_value());
  ASSERT_EQ(1, leg_fr.intermediateStops_->size());
  EXPECT_EQ("Enfant 1B", leg_fr.intermediateStops_->at(0).name_);
  EXPECT_EQ("Parent Deux", leg_fr.to_.name_);
  EXPECT_EQ("Vers Parent Deux", leg_fr.headsign_);
  EXPECT_EQ("FR_SHORT_NAME", leg_fr.routeShortName_);
  EXPECT_EQ("FR-R1", leg_fr.routeLongName_);
}

TEST(motis, trip_notice_translations) {
  auto [d, _] = get_test_case<test_case::CH_trip_notice_translations>();
  auto const day = date::sys_days{2024_y / December / 15};
  d.init_rtt(day);
  auto& rtt = *d.rt_->rtt_;

  auto const trip_ep = utl::init_from<ep::trip>(d).value();
  auto const base_trip = std::string{
      "?tripId=20241215_05%3A50_netex_ch%3A1%3AServiceJourney%3Awhatever"};

  auto const expect_notice = [&](std::optional<std::string_view> language,
                                 std::string_view expected) {
    auto url = base_trip;
    if (language.has_value()) {
      url += "&language=";
      url.append(language->data(), language->size());
    }
    auto const res = trip_ep(url);
    ASSERT_EQ(1, res.legs_.size());
    auto const& leg = res.legs_[0];
    ASSERT_TRUE(leg.alerts_.has_value());
    ASSERT_FALSE(leg.alerts_->empty());
    EXPECT_EQ(expected, leg.alerts_->front().headerText_);
  };

  expect_notice(std::nullopt, "Gratis-Internet mit der App SBB FreeSurf");
  expect_notice("de"sv, "Gratis-Internet mit der App SBB FreeSurf");
  expect_notice("en"sv, "Free Internet with the SBB FreeSurf app");
  expect_notice("fr"sv, "Connexion Internet gratuite avec l'app FreeSurf CFF");
  expect_notice("it"sv, "Connessione Internet gratuita con l'app FreeSurf FFS");

  auto const sched_dep = std::chrono::time_point_cast<std::chrono::seconds>(
      nigiri::parse_time("2024-12-15T05:50:00+01:00", "%FT%T%Ez"));
  auto const sched_arr = std::chrono::time_point_cast<std::chrono::seconds>(
      nigiri::parse_time("2024-12-15T05:57:00+01:00", "%FT%T%Ez"));
  auto const check_leg =
      [&](api::Leg const& leg, std::chrono::sys_seconds const dep,
          std::chrono::sys_seconds const arr, bool const is_rt) {
        EXPECT_EQ(dep, *leg.startTime_);
        EXPECT_EQ(arr, *leg.endTime_);
        EXPECT_EQ(sched_dep, *leg.scheduledStartTime_);
        EXPECT_EQ(sched_arr, *leg.scheduledEndTime_);
        EXPECT_EQ(is_rt, leg.realTime_);
      };

  auto const base_res = trip_ep(base_trip);
  ASSERT_EQ(1, base_res.legs_.size());
  check_leg(base_res.legs_.front(), sched_dep, sched_arr, false);

  {
    constexpr auto kNetexSiriUpdate = R"(<?xml version="1.0" encoding="UTF-8"?>
<Siri xmlns="http://www.siri.org.uk/siri" version="2.0">
  <ServiceDelivery>
    <ResponseTimestamp>2024-12-15T05:40:00</ResponseTimestamp>
    <EstimatedTimetableDelivery version="2.0">
      <ResponseTimestamp>2024-12-15T05:40:00</ResponseTimestamp>
      <EstimatedJourneyVersionFrame>
        <RecordedAtTime>2024-12-15T05:40:00</RecordedAtTime>
        <EstimatedVehicleJourney>
          <LineRef>LineDoesNotMatter</LineRef>
          <DirectionRef>Up</DirectionRef>
          <FramedVehicleJourneyRef>
            <DataFrameRef>2024-12-15</DataFrameRef>
            <DatedVehicleJourneyRef>unknown</DatedVehicleJourneyRef>
          </FramedVehicleJourneyRef>
          <RecordedCalls>
            <RecordedCall>
              <StopPointRef>ch:1:sloid:30243:0:403158</StopPointRef>
              <AimedDepartureTime>2024-12-15T05:50:00+01:00</AimedDepartureTime>
              <ExpectedDepartureTime>2024-12-15T05:52:00+01:00</ExpectedDepartureTime>
            </RecordedCall>
          </RecordedCalls>
          <EstimatedCalls>
            <EstimatedCall>
              <StopPointRef>ch:1:sloid:1954:0:845083</StopPointRef>
              <AimedArrivalTime>2024-12-15T05:57:00+01:00</AimedArrivalTime>
              <ExpectedArrivalTime>2024-12-15T05:59:00+01:00</ExpectedArrivalTime>
            </EstimatedCall>
          </EstimatedCalls>
        </EstimatedVehicleJourney>
      </EstimatedJourneyVersionFrame>
    </EstimatedTimetableDelivery>
  </ServiceDelivery>
</Siri>
)";

    auto siri_updater = auser(*d.tt_, d.tags_->get_src("netex"),
                              nigiri::rt::vdv_aus::updater::xml_format::kSiri);
    auto const expected_siri_state =
        nigiri::parse_time_no_tz("2024-12-15T05:40:00")
            .time_since_epoch()
            .count();
    auto const siri_stats = siri_updater.consume_update(kNetexSiriUpdate, rtt);
    EXPECT_EQ(1U, siri_stats.matched_runs_);
    EXPECT_EQ(2U, siri_stats.updated_events_);
    EXPECT_EQ(expected_siri_state, siri_updater.update_state_);
    auto const siri_dep = std::chrono::time_point_cast<std::chrono::seconds>(
        nigiri::parse_time("2024-12-15T05:52:00+01:00", "%FT%T%Ez"));
    auto const siri_arr = std::chrono::time_point_cast<std::chrono::seconds>(
        nigiri::parse_time("2024-12-15T05:59:00+01:00", "%FT%T%Ez"));
    auto const siri_res = trip_ep(base_trip);
    ASSERT_EQ(1, siri_res.legs_.size());
    check_leg(siri_res.legs_.front(), siri_dep, siri_arr, true);
  }

  {
    constexpr auto kNetexVdvUpdate =
        R"(<?xml version="1.0" encoding="iso-8859-1"?>
<DatenAbrufenAntwort>
  <Bestaetigung Zst="2024-12-15T05:40:00" Ergebnis="ok" Fehlernummer="0" />
  <AUSNachricht AboID="1" auser_id="314159">
    <IstFahrt Zst="2024-12-15T05:50:00">
      <LinienID>NET</LinienID>
      <RichtungsID>1</RichtungsID>
      <FahrtRef>
        <FahrtID>
          <FahrtBezeichner>NETEX</FahrtBezeichner>
          <Betriebstag>2024-12-15</Betriebstag>
        </FahrtID>
      </FahrtRef>
      <Komplettfahrt>true</Komplettfahrt>
      <BetreiberID>MOTIS</BetreiberID>
      <IstHalt>
        <HaltID>ch:1:sloid:30243:0:403158</HaltID>
        <Abfahrtszeit>2024-12-15T04:50:00</Abfahrtszeit>
        <IstAbfahrtPrognose>2024-12-15T04:56:00</IstAbfahrtPrognose>
      </IstHalt>
      <IstHalt>
        <HaltID>ch:1:sloid:1954:0:845083</HaltID>
        <Ankunftszeit>2024-12-15T04:57:00</Ankunftszeit>
        <IstAnkunftPrognose>2024-12-15T05:03:00</IstAnkunftPrognose>
      </IstHalt>
      <LinienText>NET</LinienText>
      <ProduktID>NET</ProduktID>
      <RichtungsText>Netex Demo</RichtungsText>
      <Zusatzfahrt>false</Zusatzfahrt>
      <FaelltAus>false</FaelltAus>
    </IstFahrt>
  </AUSNachricht>
</DatenAbrufenAntwort>
)";

    auto vdv_updater = auser(*d.tt_, d.tags_->get_src("netex"),
                             nigiri::rt::vdv_aus::updater::xml_format::kVdv);
    auto const vdv_stats = vdv_updater.consume_update(kNetexVdvUpdate, rtt);
    EXPECT_EQ(1U, vdv_stats.matched_runs_);
    EXPECT_EQ(2U, vdv_stats.updated_events_);
    EXPECT_EQ(314159, vdv_updater.update_state_);
    auto const vdv_dep = std::chrono::time_point_cast<std::chrono::seconds>(
        nigiri::parse_time("2024-12-15T04:56:00", "%FT%T"));
    auto const vdv_arr = std::chrono::time_point_cast<std::chrono::seconds>(
        nigiri::parse_time("2024-12-15T05:03:00", "%FT%T"));
    auto const vdv_res = trip_ep(base_trip);
    ASSERT_EQ(1, vdv_res.legs_.size());
    check_leg(vdv_res.legs_.front(), vdv_dep, vdv_arr, true);
  }
}
