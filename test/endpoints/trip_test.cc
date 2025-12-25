#include "gtest/gtest.h"

#include "utl/init_from.h"

#include "nigiri/common/parse_time.h"
#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/frun.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/trip.h"
#include "motis/import.h"
#include "motis/rt/auser.h"
#include "motis/tag_lookup.h"

using namespace std::string_view_literals;
using namespace motis;
using namespace date;

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
Parent1,Parent1,50.0,8.0,1,,
Child1A,Child1A,50.001,8.001,0,Parent1,1
Child1B,Child1B,50.002,8.002,0,Parent1,2
Parent2,Parent2,51.0,9.0,1,,
Child2,Child2,51.001,9.001,0,Parent2,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,DB,R1,R1,,109
R2,DB,R2,R2,,109

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,Parent2 Express,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type,stop_headsign
T1,10:00:00,10:00:00,Child1A,1,0,0,Origin
T1,10:10:00,10:10:00,Child1B,2,0,0,Midway
T1,11:00:00,11:00:00,Child2,3,0,0,Destination

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# translations.txt
table_name,field_name,language,translation,record_id,record_sub_id,field_value
routes,route_long_name,de,DE-R1,,,R1
routes,route_long_name,fr,FR-R1,,,R1
routes,route_long_name,en,EN-R1,,,R1
stops,stop_name,en,Child1A,Child1A,,
stops,stop_name,de,Kind 1A,Child1A,,
stops,stop_name,en,Child1B,,,Child1B
stops,stop_name,de,Kind 1B,,,Child1B
stops,stop_name,en,Parent2,Parent2,,
stops,stop_name,de,Eltern 2,Parent2,,
stops,stop_name,fr,Parent Deux,Parent2,,
stops,stop_name,fr,Enfant 1A,Child1A,,
stops,stop_name,fr,Enfant 1B,,,Child1B
stop_times,stop_headsign,en,Parent2 Express,T1,1,
stop_times,stop_headsign,de,Richtung Eltern Zwei,T1,1,
stop_times,stop_headsign,fr,Vers Parent Deux,T1,1,
)";

constexpr auto kScript = R"(
function process_route(route)
  route:set_short_name({
    translation.new('en', 'EN_SHORT_NAME'),
    translation.new('de', 'DE_SHORT_NAME'),
    translation.new('fr', 'FR_SHORT_NAME')
  })
  route:get_short_name_translations():add(translation.new('hu', 'HU_SHORT_NAME'))
  print(route:get_short_name_translations():get(1):get_text())
  print(route:get_short_name_translations():get(1):get_language())
end
)";

TEST(motis, trip_stop_naming) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .timetable_ =
          config::timetable{
              .first_day_ = "2019-05-01",
              .num_days_ = 2,
              .datasets_ = {{"test", {.path_ = kGTFS, .script_ = kScript}}}},
      .street_routing_ = false};
  auto d = import(c, "test/data", true);

  auto const trip_ep = utl::init_from<ep::trip>(d).value();

  auto const res = trip_ep("?tripId=20190501_10%3A00_test_T1");
  ASSERT_EQ(1, res.legs_.size());
  auto const& leg = res.legs_[0];
  EXPECT_EQ("Child1A", leg.from_.name_);
  ASSERT_TRUE(leg.intermediateStops_.has_value());
  ASSERT_EQ(1, leg.intermediateStops_->size());
  EXPECT_EQ("Child1B", leg.intermediateStops_->at(0).name_);
  EXPECT_EQ("Parent2", leg.to_.name_);
  EXPECT_EQ("Parent2 Express", leg.headsign_);
  EXPECT_EQ("EN_SHORT_NAME", leg.routeShortName_);
  EXPECT_EQ("EN-R1", leg.routeLongName_);

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

constexpr auto kNetex = R"(
# netex.xml
<?xml version="1.0" encoding="UTF-8"?>
<PublicationDelivery xmlns:gml="http://www.opengis.net/gml/3.2"
                     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                     xmlns:siri="http://www.siri.org.uk/siri"
                     xmlns="http://www.netex.org.uk/netex"
                     version="1.09"
                     xsi:schemaLocation="http://www.netex.org.uk/netex http://netex.uk/netex/schema/1.10/xsd/NeTEx_publication.xsd">
  <PublicationTimestamp>2025-06-26T14:16:54+02:00</PublicationTimestamp>
  <ParticipantRef>INTERMAPS</ParticipantRef>
  <dataObjects>
    <CompositeFrame id="ch:1:CompositeFrame:intermaps" version="any">
      <FrameDefaults>
        <DefaultLocale>
          <TimeZoneOffset>1</TimeZoneOffset>
          <SummerTimeZoneOffset>2</SummerTimeZoneOffset>
          <DefaultLanguage>de</DefaultLanguage>
        </DefaultLocale>
      </FrameDefaults>
      <ValidBetween>
        <FromDate>2024-12-15T00:00:00</FromDate>
        <ToDate>2025-12-14T23:59:59</ToDate>
      </ValidBetween>
      <frames>
        <ResourceFrame id="ch:1:ResourceFrame:1" version="any">
          <typesOfValue>
            <ValueSet id="ch:1:ValueSet:TypeOfProductCategory" version="any" nameOfClass="TypeOfProductCategory">
              <Name>ProductCategories</Name>
              <values>
                <TypeOfProductCategory id="ch:1:TypeOfProductCategory:PB" version="any">
                  <Name lang="de">Pendelbahn</Name>
                  <ShortName lang="de">PB</ShortName>
                </TypeOfProductCategory>
              </values>
            </ValueSet>
          </typesOfValue>
          <vehicleTypes>
            <VehicleType id="ch:1:VehicleType:cabin" version="any">
              <Name>Cabin</Name>
              <ShortName>CBN</ShortName>
            </VehicleType>
          </vehicleTypes>
          <organisations>
            <Operator id="ch:1:sboid:100220" version="any">
              <PublicCode>PB</PublicCode>
              <Name>Test Operator</Name>
            </Operator>
          </organisations>
        </ResourceFrame>
        <ServiceCalendarFrame id="ch:1:ServiceCalendarFrame:ts3" version="any">
          <validityConditions>
            <AvailabilityCondition id="ch:1:AvailabilityCondition:whatever" version="any">
              <FromDate>2024-12-15T00:00:00</FromDate>
              <ToDate>2024-12-15T23:59:59</ToDate>
              <ValidDayBits>1</ValidDayBits>
            </AvailabilityCondition>
          </validityConditions>
        </ServiceCalendarFrame>
        <SiteFrame id="ch:1:SiteFrame:1" version="any">
          <stopPlaces>
            <StopPlace id="ch:1:StopPlace:30243" version="any">
              <keyList>
                <KeyValue>
                  <Key>SLOID</Key>
                  <Value>ch:1:sloid:30243</Value>
                </KeyValue>
              </keyList>
              <Name>Bettmeralp Talstation (Seilb.)</Name>
              <Centroid>
                <Location>
                  <Longitude>8.1967</Longitude>
                  <Latitude>46.3803</Latitude>
                </Location>
              </Centroid>
              <quays>
                <Quay id="ch:1:Quay:30243:1" version="any">
                  <keyList>
                    <KeyValue>
                      <Key>SLOID</Key>
                      <Value>ch:1:sloid:30243:0:403158</Value>
                    </KeyValue>
                  </keyList>
                  <Name>Bettmeralp Talstation (Seilb.)</Name>
                  <Centroid>
                    <Location>
                      <Longitude>8.1967</Longitude>
                      <Latitude>46.3803</Latitude>
                    </Location>
                  </Centroid>
                </Quay>
              </quays>
            </StopPlace>
            <StopPlace id="ch:1:StopPlace:1954" version="any">
              <keyList>
                <KeyValue>
                  <Key>SLOID</Key>
                  <Value>ch:1:sloid:1954</Value>
                </KeyValue>
              </keyList>
              <Name>Bettmeralp</Name>
              <Centroid>
                <Location>
                  <Longitude>8.1977</Longitude>
                  <Latitude>46.4219</Latitude>
                </Location>
              </Centroid>
              <quays>
                <Quay id="ch:1:Quay:1954:1" version="any">
                  <keyList>
                    <KeyValue>
                      <Key>SLOID</Key>
                      <Value>ch:1:sloid:1954:0:845083</Value>
                    </KeyValue>
                  </keyList>
                  <Name>Bettmeralp</Name>
                  <Centroid>
                    <Location>
                      <Longitude>8.1977</Longitude>
                      <Latitude>46.4219</Latitude>
                    </Location>
                  </Centroid>
                </Quay>
              </quays>
            </StopPlace>
          </stopPlaces>
        </SiteFrame>
        <ServiceFrame id="ch:1:ServiceFrame:ts3" version="any">
          <lines>
            <Line id="ch:1:slnid:1024859" version="any">
              <Name>2336 - Betten Talstation - Bettmeralp (Direkt)</Name>
              <TransportMode>bus</TransportMode>
              <TransportSubmode>
                <BusSubmode>localBus</BusSubmode>
              </TransportSubmode>
              <TypeOfProductCategoryRef ref="ch:1:TypeOfProductCategory:PB" version="any" />
              <additionalOperators>
                <OperatorRef ref="ch:1:sboid:100220" version="any" />
              </additionalOperators>
            </Line>
          </lines>
          <destinationDisplays>
            <DestinationDisplay id="ch:1:DestinationDisplay:alp" version="any">
              <FrontText>Bettmeralp</FrontText>
            </DestinationDisplay>
          </destinationDisplays>
          <scheduledStopPoints>
            <ScheduledStopPoint id="ch:1:sloid:30243:0:403158" version="any">
              <Name lang="de">Bettmeralp Talstation (Seilb.)</Name>
            </ScheduledStopPoint>
            <ScheduledStopPoint id="ch:1:sloid:1954:0:845083" version="any">
              <Name lang="de">Bettmeralp</Name>
            </ScheduledStopPoint>
          </scheduledStopPoints>
          <stopAssignments>
            <PassengerStopAssignment id="ch:1:PassengerStopAssignment:1" version="any">
              <ScheduledStopPointRef ref="ch:1:sloid:30243:0:403158" version="any" />
              <QuayRef ref="ch:1:Quay:30243:1" version="any" />
            </PassengerStopAssignment>
            <PassengerStopAssignment id="ch:1:PassengerStopAssignment:2" version="any">
              <ScheduledStopPointRef ref="ch:1:sloid:1954:0:845083" version="any" />
              <QuayRef ref="ch:1:Quay:1954:1" version="any" />
            </PassengerStopAssignment>
          </stopAssignments>
          <notices>
            <Notice id="ch:1:Notice:A__FS" version="any">
              <alternativeTexts>
                <AlternativeText attributeName="Text">
                  <Text lang="en">Free Internet with the SBB FreeSurf app</Text>
                </AlternativeText>
                <AlternativeText attributeName="Text">
                  <Text lang="fr">Connexion Internet gratuite avec l'app FreeSurf CFF</Text>
                </AlternativeText>
                <AlternativeText attributeName="Text">
                  <Text lang="it">Connessione Internet gratuita con l'app FreeSurf FFS</Text>
                </AlternativeText>
              </alternativeTexts>
              <Text lang="de">Gratis-Internet mit der App SBB FreeSurf</Text>
              <ShortCode>A__FS</ShortCode>
              <PrivateCode>A__FS</PrivateCode>
              <TypeOfNoticeRef ref="ch:1:TypeOfNotice:10" version="any" />
              <CanBeAdvertised>true</CanBeAdvertised>
            </Notice>
          </notices>
        </ServiceFrame>
        <TimetableFrame id="ch:1:TimetableFrame:ts3" version="any">
          <vehicleJourneys>
            <ServiceJourney id="ch:1:ServiceJourney:whatever" version="any">
              <keyList>
                <KeyValue>
                  <Key>TripNr</Key>
                  <Value>2336</Value>
                </KeyValue>
              </keyList>
              <validityConditions>
                <AvailabilityConditionRef ref="ch:1:AvailabilityCondition:whatever" version="any" />
              </validityConditions>
              <TypeOfProductCategoryRef ref="ch:1:TypeOfProductCategory:PB" version="any" />
              <DepartureTime>05:50:00</DepartureTime>
              <JourneyDuration>PT7M</JourneyDuration>
              <OperatorRef ref="ch:1:sboid:100220" version="any" />
              <VehicleTypeRef ref="ch:1:VehicleType:cabin" version="any" />
              <LineRef ref="ch:1:slnid:1024859" version="any" />
              <DirectionType>inbound</DirectionType>
              <calls>
                <Call id="ch:1:Call:whatever1" version="any" order="1">
                  <ScheduledStopPointRef ref="ch:1:sloid:30243:0:403158" version="any" />
                  <Departure>
                    <Time>05:50:00</Time>
                  </Departure>
                  <DestinationDisplayRef ref="ch:1:DestinationDisplay:alp" version="any" />
                  <noticeAssignments>
                    <NoticeAssignment id="ch:1:NoticeAssignment:1" order="1">
                      <NoticeRef ref="ch:1:Notice:A__FS" version="any" />
                    </NoticeAssignment>
                  </noticeAssignments>
                </Call>
                <Call id="ch:1:Call:whatever2" version="any" order="2">
                  <ScheduledStopPointRef ref="ch:1:sloid:1954:0:845083" version="any" />
                  <Arrival>
                    <Time>05:57:00</Time>
                  </Arrival>
                  <DestinationDisplayRef ref="ch:1:DestinationDisplay:alp" version="any" />
                  <noticeAssignments>
                    <NoticeAssignment id="ch:1:NoticeAssignment:2" order="1">
                      <NoticeRef ref="ch:1:Notice:A__FS" version="any" />
                    </NoticeAssignment>
                  </noticeAssignments>
                </Call>
              </calls>
            </ServiceJourney>
          </vehicleJourneys>
        </TimetableFrame>
      </frames>
    </CompositeFrame>
  </dataObjects>
</PublicationDelivery>
)";

TEST(motis, trip_notice_translations) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2024-12-15",
                                   .num_days_ = 2,
                                   .datasets_ = {{"netex", {.path_ = kNetex}}}},
             .street_routing_ = false};
  auto d = import(c, "test/data", true);
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
