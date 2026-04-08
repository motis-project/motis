#include "motis/config.h"

#include "../test_case.h"

using motis::config;

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

template <>
test_case_params const import_test_case<test_case::CH_cable_car_netex>() {
  auto const c =
      config{.timetable_ =
                 config::timetable{.first_day_ = "2024-12-15",
                                   .num_days_ = 2,
                                   .datasets_ = {{"netex", {.path_ = kNetex}}}},
             .street_routing_ = false};
  return import_test_case(std::move(c),
                          "test/test_case/CH_cable_car_netex_data");
}
