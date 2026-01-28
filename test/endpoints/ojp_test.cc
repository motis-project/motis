#include "gtest/gtest.h"

#include <filesystem>
#include <string>
#include <string_view>

#include "date/date.h"

#include "utl/init_from.h"

#include "adr/formatter.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/endpoints/ojp.h"
#include "motis/import.h"

using namespace motis;
using namespace date;

constexpr auto const kOjpGtfs = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
TEST,Test Agency,https://example.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,Darmstadt Hauptbahnhof,49.8726,8.63085,1,,
DA_3,Darmstadt Hauptbahnhof,49.87355,8.63003,0,DA,3
8507000,Stop 8507000,49.87200,8.62900,1,,
8507000_1,Stop 8507000,49.87200,8.62900,0,8507000,1

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
R1,TEST,R1,,,3

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
R1,S1,T1,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
T1,07:00:00,07:00:00,8507000_1,1,0,0
T1,07:30:00,07:30:00,DA_3,2,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20260128,1
)";

constexpr auto const kOjpGeocodingRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2026-01-28T07:22:00.862Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPLocationInformationRequest>
      <siri:RequestTimestamp>2026-01-28T07:22:00.862Z</siri:RequestTimestamp>
      <InitialInput>
        <Name>Darmstadt Hauptbahnhof</Name>
      </InitialInput>
      <Restrictions>
        <Type>stop</Type>
        <NumberOfResults>10</NumberOfResults>
        <IncludePtModes>true</IncludePtModes>
      </Restrictions>
    </OJPLocationInformationRequest>
  </siri:ServiceRequest>
</OJPRequest>

</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpMapStopsRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2026-01-28T07:47:11.377Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPLocationInformationRequest>
      <siri:RequestTimestamp>2026-01-28T07:47:11.377Z</siri:RequestTimestamp>
      <InitialInput>
        <GeoRestriction>
          <Rectangle>
            <UpperLeft>
              <siri:Latitude>49.87400</siri:Latitude>
              <siri:Longitude>8.62850</siri:Longitude>
            </UpperLeft>
            <LowerRight>
              <siri:Latitude>49.87100</siri:Latitude>
              <siri:Longitude>8.63250</siri:Longitude>
            </LowerRight>
          </Rectangle>
        </GeoRestriction>
      </InitialInput>
      <Restrictions>
        <Type>stop</Type>
        <NumberOfResults>300</NumberOfResults>
        <IncludePtModes>true</IncludePtModes>
      </Restrictions>
    </OJPLocationInformationRequest>
  </siri:ServiceRequest>
</OJPRequest>

</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpStopEventRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2026-01-28T07:38:53.987Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPStopEventRequest>
      <siri:RequestTimestamp>2026-01-28T07:38:53.987Z</siri:RequestTimestamp>
      <Location>
        <PlaceRef>
          <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
        <DepArrTime>2026-01-28T06:00:00.000Z</DepArrTime>
      </Location>
      <Params>
        <IncludeAllRestrictedLines>true</IncludeAllRestrictedLines>
        <NumberOfResults>10</NumberOfResults>
        <StopEventType>departure</StopEventType>
        <IncludePreviousCalls>true</IncludePreviousCalls>
        <IncludeOnwardCalls>true</IncludeOnwardCalls>
        <UseRealtimeData>explanatory</UseRealtimeData>
      </Params>
    </OJPStopEventRequest>
  </siri:ServiceRequest>
</OJPRequest>

</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpTripInfoRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2026-01-28T07:42:51.310Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPTripInfoRequest>
      <siri:RequestTimestamp>2026-01-28T07:42:51.310Z</siri:RequestTimestamp>
      <JourneyRef>20260128_07:00_test_T1</JourneyRef>
      <OperatingDayRef>2026-01-28</OperatingDayRef>
      <Params>
        <IncludeCalls>true</IncludeCalls>
        <IncludeService>true</IncludeService>
        <IncludeTrackProjection>true</IncludeTrackProjection>
        <IncludePlacesContext>true</IncludePlacesContext>
        <IncludeSituationsContext>true</IncludeSituationsContext>
      </Params>
    </OJPTripInfoRequest>
  </siri:ServiceRequest>
</OJPRequest>

</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpRoutingRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2026-01-28T07:54:19.394Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPTripRequest>
      <siri:RequestTimestamp>2026-01-28T07:54:19.394Z</siri:RequestTimestamp>
      <Origin>
        <PlaceRef>
          <StopPlaceRef>test_8507000</StopPlaceRef>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
        <DepArrTime>2026-01-28T06:00:00.000Z</DepArrTime>
        <IndividualTransportOption />
      </Origin>
      <Destination>
        <PlaceRef>
          <StopPlaceRef>test_DA</StopPlaceRef>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
        <IndividualTransportOption />
      </Destination>
      <Params>
        <NumberOfResults>5</NumberOfResults>
        <IncludeAllRestrictedLines>true</IncludeAllRestrictedLines>
        <IncludeTrackSections>true</IncludeTrackSections>
        <IncludeLegProjection>false</IncludeLegProjection>
        <IncludeTurnDescription>true</IncludeTurnDescription>
        <IncludeIntermediateStops>true</IncludeIntermediateStops>
        <UseRealtimeData>explanatory</UseRealtimeData>
      </Params>
    </OJPTripRequest>
  </siri:ServiceRequest>
</OJPRequest>
</OJP>
)";

constexpr auto const kOjpGeocodingResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>2026-01-28T07:22:00.862Z</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>1</siri:ResponseMessageIdentifier>
      <OJPLocationInformationDelivery>
        <siri:ResponseTimestamp>2026-01-28T07:22:00.862Z</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_DA</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
            </Name>
            <GeoPosition>
              <siri:Latitude>49.8726</siri:Latitude>
              <siri:Longitude>8.63085</siri:Longitude>
            </GeoPosition>
            <Mode>
              <PtMode>bus</PtMode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
      </OJPLocationInformationDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

constexpr auto const kOjpMapStopsResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>2026-01-28T07:47:11.377Z</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>2</siri:ResponseMessageIdentifier>
      <OJPLocationInformationDelivery>
        <siri:ResponseTimestamp>2026-01-28T07:47:11.377Z</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_DA</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
            </Name>
            <GeoPosition>
              <siri:Latitude>49.8726</siri:Latitude>
              <siri:Longitude>8.63085</siri:Longitude>
            </GeoPosition>
            <Mode>
              <PtMode>bus</PtMode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_DA_3</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
            </Name>
            <GeoPosition>
              <siri:Latitude>49.8736</siri:Latitude>
              <siri:Longitude>8.63003</siri:Longitude>
            </GeoPosition>
            <Mode>
              <PtMode>bus</PtMode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_8507000</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">Stop 8507000</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">Stop 8507000</Text>
            </Name>
            <GeoPosition>
              <siri:Latitude>49.872</siri:Latitude>
              <siri:Longitude>8.629</siri:Longitude>
            </GeoPosition>
            <Mode>
              <PtMode>bus</PtMode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_8507000_1</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">Stop 8507000</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">Stop 8507000</Text>
            </Name>
            <GeoPosition>
              <siri:Latitude>49.872</siri:Latitude>
              <siri:Longitude>8.629</siri:Longitude>
            </GeoPosition>
            <Mode>
              <PtMode>bus</PtMode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
      </OJPLocationInformationDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

[[maybe_unused]] constexpr auto const kOjpStopEventResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>2026-01-28T07:38:53.987Z</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>3</siri:ResponseMessageIdentifier>
      <OJPStopEventDelivery>
        <siri:ResponseTimestamp>2026-01-28T07:38:53.987Z</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <StopEventResponseContext>
          <Places>
            <Place>
              <Name>Stop 8507000</Name>
              <GeoPosition>
                <siri:Latitude>49.872</siri:Latitude>
                <siri:Longitude>8.629</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_8507000</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPointName>
                <ParentRef>test_8507000</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Stop 8507000</Name>
              <GeoPosition>
                <siri:Latitude>49.872</siri:Latitude>
                <siri:Longitude>8.629</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPointName>
                <ParentRef>test_8507000</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Darmstadt Hauptbahnhof</Name>
              <GeoPosition>
                <siri:Latitude>49.8726</siri:Latitude>
                <siri:Longitude>8.63085</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_DA</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Darmstadt Hauptbahnhof</Name>
              <GeoPosition>
                <siri:Latitude>49.8736</siri:Latitude>
                <siri:Longitude>8.63003</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
            </Place>
          </Places>
          <Situations />
        </StopEventResponseContext>
        <StopEventResult>
          <Id>1</Id>
          <StopEvent>
            <ThisCall>
              <CallAtStop>
                <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                <StopPointName>Stop 8507000</StopPointName>
                <PlannedQuay>1</PlannedQuay>
                <ServiceDeparture>
                  <TimetabledTime>2026-01-28T06:00:00Z</TimetabledTime>
                  <EstimatedTime>2026-01-28T06:00:00Z</EstimatedTime>
                </ServiceDeparture>
                <Order>1</Order>
              </CallAtStop>
            </ThisCall>
            <OnwardCall>
              <CallAtStop>
                <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                <StopPointName>Darmstadt Hauptbahnhof</StopPointName>
                <PlannedQuay>3</PlannedQuay>
                <ServiceArrival>
                  <TimetabledTime>2026-01-28T06:30:00Z</TimetabledTime>
                  <EstimatedTime>2026-01-28T06:30:00Z</EstimatedTime>
                </ServiceArrival>
                <Order>2</Order>
              </CallAtStop>
            </OnwardCall>
            <Service>
              <OperatingDayRef>20260128</OperatingDayRef>
              <JourneyRef>20260128_07:00_test_T1</JourneyRef>
              <PublicCode>R1</PublicCode>
              <PublicCode>R1</PublicCode>
              <siri:LineRef>R1</siri:LineRef>
              <siri:DirectionRef>0</siri:DirectionRef>
              <Mode>
                <PtMode>bus</PtMode>
                <Name>
                  <Text xml:lang="de">R1</Text>
                </Name>
                <ShortName>
                  <Text xml:lang="de">R1</Text>
                </ShortName>
              </Mode>
              <PublishedServiceName>
                <Text xml:lang="de">R1</Text>
              </PublishedServiceName>
              <TrainNumber></TrainNumber>
              <OriginText>
                <Text xml:lang="de">n/a</Text>
              </OriginText>
              <siri:OperatorRef>TEST</siri:OperatorRef>
              <DestinationText>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </DestinationText>
            </Service>
          </StopEvent>
        </StopEventResult>
      </OJPStopEventDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

[[maybe_unused]] constexpr auto const kOjpTripInfoResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>2026-01-28T07:42:51.310Z</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>4</siri:ResponseMessageIdentifier>
      <OJPTripInfoDelivery>
        <siri:ResponseTimestamp>2026-01-28T07:42:51.310Z</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <TripInfoResponseContext>
          <Places>
            <Place>
              <Name>Stop 8507000</Name>
              <GeoPosition>
                <siri:Latitude>49.872</siri:Latitude>
                <siri:Longitude>8.629</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_8507000</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPointName>
                <ParentRef>test_8507000</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Stop 8507000</Name>
              <GeoPosition>
                <siri:Latitude>49.872</siri:Latitude>
                <siri:Longitude>8.629</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPointName>
                <ParentRef>test_8507000</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Darmstadt Hauptbahnhof</Name>
              <GeoPosition>
                <siri:Latitude>49.8726</siri:Latitude>
                <siri:Longitude>8.63085</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_DA</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
            </Place>
            <Place>
              <Name>Darmstadt Hauptbahnhof</Name>
              <GeoPosition>
                <siri:Latitude>49.8736</siri:Latitude>
                <siri:Longitude>8.63003</siri:Longitude>
              </GeoPosition>
              <StopPoint>
                <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
            </Place>
          </Places>
          <Situations />
        </TripInfoResponseContext>
        <TripInfoResult>
          <PreviousCall>
            <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
            <StopPointName>Stop 8507000</StopPointName>
            <PlannedQuay>1</PlannedQuay>
            <ServiceArrival />
            <ServiceDeparture>
              <TimetabledTime>2026-01-28T06:00:00Z</TimetabledTime>
              <EstimatedTime>2026-01-28T06:00:00Z</EstimatedTime>
            </ServiceDeparture>
            <Order>1</Order>
          </PreviousCall>
          <PreviousCall>
            <siri:StopPointRef>test_DA_3</siri:StopPointRef>
            <StopPointName>Darmstadt Hauptbahnhof</StopPointName>
            <PlannedQuay>3</PlannedQuay>
            <ServiceArrival>
              <TimetabledTime>2026-01-28T06:30:00Z</TimetabledTime>
              <EstimatedTime>2026-01-28T06:30:00Z</EstimatedTime>
            </ServiceArrival>
            <ServiceDeparture />
            <Order>2</Order>
          </PreviousCall>
          <Service>
            <OperatingDayRef>2026-01-28</OperatingDayRef>
            <JourneyRef>20260128_07:00_test_T1</JourneyRef>
            <PublicCode>R1</PublicCode>
            <PublicCode>R1</PublicCode>
            <siri:LineRef>R1</siri:LineRef>
            <siri:DirectionRef>0</siri:DirectionRef>
            <Mode>
              <PtMode>bus</PtMode>
              <Name>
                <Text xml:lang="de">R1</Text>
              </Name>
              <ShortName>
                <Text xml:lang="de">R1</Text>
              </ShortName>
            </Mode>
            <PublishedServiceName>
              <Text xml:lang="de">R1</Text>
            </PublishedServiceName>
            <TrainNumber></TrainNumber>
            <OriginText>
              <Text xml:lang="de">n/a</Text>
            </OriginText>
            <siri:OperatorRef>TEST</siri:OperatorRef>
            <DestinationText>
              <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
            </DestinationText>
          </Service>
          <JourneyTrack>
            <TrackSection>
              <TrackSectionStart>
                <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                <Name>Stop 8507000</Name>
              </TrackSectionStart>
              <TrackSectionEnd>
                <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                <Name>Darmstadt Hauptbahnhof</Name>
              </TrackSectionEnd>
              <LinkProjection>
                <Position>
                  <siri:Latitude>498.72</siri:Latitude>
                  <siri:Longitude>86.29</siri:Longitude>
                </Position>
                <Position>
                  <siri:Latitude>498.736</siri:Latitude>
                  <siri:Longitude>86.3003</siri:Longitude>
                </Position>
              </LinkProjection>
              <Duration>PT30M</Duration>
              <Length>0</Length>
            </TrackSection>
          </JourneyTrack>
        </TripInfoResult>
      </OJPTripInfoDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

[[maybe_unused]] constexpr auto const kOjpRoutingResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>2026-01-28T07:54:19.394Z</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>ojp-routing-1</siri:ResponseMessageIdentifier>
      <OJPTripDelivery>
        <siri:ResponseTimestamp>2026-01-28T07:54:19.394Z</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <TripResponseContext>
          <Places>
            <Place>
              <StopPlace>
                <StopPlaceRef>test_8507000</StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">Stop 8507000</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.62900</siri:Longitude>
                <siri:Latitude>49.87200</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Stop 8507000</Text>
                </StopPointName>
                <ParentRef>test_8507000</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">Stop 8507000</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.62900</siri:Longitude>
                <siri:Latitude>49.87200</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPlace>
                <StopPlaceRef>test_DA</StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.63085</siri:Longitude>
                <siri:Latitude>49.8726</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.63003</siri:Longitude>
                <siri:Latitude>49.87355</siri:Latitude>
              </GeoPosition>
            </Place>
          </Places>
        </TripResponseContext>
        <TripResult>
          <Id>ojp-trip-1</Id>
          <Trip>
            <Id>ojp-trip-1</Id>
            <Duration>PT30M</Duration>
            <StartTime>2026-01-28T06:00:00Z</StartTime>
            <EndTime>2026-01-28T06:30:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>0</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT30M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_8507000_1</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">Stop 8507000</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">1</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2026-01-28T06:00:00Z</TimetabledTime>
                    <EstimatedTime>2026-01-28T06:00:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_DA_3</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">Darmstadt Hauptbahnhof</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">3</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2026-01-28T06:30:00Z</TimetabledTime>
                    <EstimatedTime>2026-01-28T06:30:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>2026-01-28</OperatingDayRef>
                  <JourneyRef>20260128_07:00_test_T1</JourneyRef>
                  <PublishedServiceName>
                    <Text xml:lang="de">R1</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>bus</PtMode>
                  </Mode>
                </Service>
              </TimedLeg>
            </Leg>
          </Trip>
        </TripResult>
      </OJPTripDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

TEST(motis, ojp_requests) {
  auto ec = std::error_code{};
  std::filesystem::remove_all("test/data", ec);

  auto const c = config{
      .timetable_ =
          config::timetable{.first_day_ = "2026-01-28",
                            .num_days_ = 1,
                            .datasets_ = {{"test", {.path_ = kOjpGtfs}}}},
      .geocoding_ = true};
  import(c, "test/data", true);
  auto d = data{"test/data", c};
  d.init_rtt(sys_days{2026_y / January / 28});

  auto const ojp_ep = ep::ojp{
      .routing_ep_ = utl::init_from<ep::routing>(d),
      .geocoding_ep_ = utl::init_from<ep::geocode>(d),
      .stops_ep_ = utl::init_from<ep::stops>(d),
      .stop_times_ep_ = utl::init_from<ep::stop_times>(d),
      .trip_ep_ = utl::init_from<ep::trip>(d),
  };

  auto const send_request = [&](std::string_view body) {
    net::web_server::http_req_t req{boost::beast::http::verb::post,
                                    "/api/v2/ojp", 11};
    req.set(boost::beast::http::field::content_type, "text/xml; charset=utf-8");
    req.body() = std::string{body};
    req.prepare_payload();
    return ojp_ep(net::route_request{std::move(req)}, false);
  };

  auto const normalize_response_timestamps = [](std::string_view input) {
    auto out = std::string{input};
    auto constexpr kStart = std::string_view{"<siri:ResponseTimestamp>"};
    auto constexpr kEnd = std::string_view{"</siri:ResponseTimestamp>"};
    auto pos = std::size_t{0};
    while ((pos = out.find(kStart, pos)) != std::string::npos) {
      auto const value_start = pos + kStart.size();
      auto const value_end = out.find(kEnd, value_start);
      if (value_end == std::string::npos) {
        break;
      }
      out.replace(value_start, value_end - value_start, "NOW");
      pos = value_end + kEnd.size();
    }
    return out;
  };

  auto const expect_response = [&](net::reply const& reply,
                                   std::string_view expected) {
    auto const* res = std::get_if<net::web_server::string_res_t>(&reply);
    ASSERT_NE(nullptr, res);
    EXPECT_EQ(boost::beast::http::status::ok, res->result());
    EXPECT_EQ("text/xml; charset=utf-8",
              res->base()[boost::beast::http::field::content_type]);
    EXPECT_EQ(normalize_response_timestamps(expected),
              normalize_response_timestamps(res->body()));
  };

  expect_response(send_request(kOjpGeocodingRequest), kOjpGeocodingResponse);
  expect_response(send_request(kOjpMapStopsRequest), kOjpMapStopsResponse);
  expect_response(send_request(kOjpStopEventRequest), kOjpStopEventResponse);
  expect_response(send_request(kOjpTripInfoRequest), kOjpTripInfoResponse);
  // expect_response(send_request(kOjpRoutingRequest), kOjpRoutingResponse);
}
