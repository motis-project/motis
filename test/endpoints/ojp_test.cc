#include "gtest/gtest.h"

#include <filesystem>
#include <fstream>
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

constexpr auto const kGTFS = R"(
# agency.txt
agency_id,agency_name,agency_url,agency_timezone
DB,Deutsche Bahn,https://deutschebahn.com,Europe/Berlin

# stops.txt
stop_id,stop_name,stop_lat,stop_lon,location_type,parent_station,platform_code
DA,DA Hbf,49.87260,8.63085,1,,
DA_3,DA Hbf,49.87355,8.63003,0,DA,3
DA_10,DA Hbf,49.87336,8.62926,0,DA,10
FFM,FFM Hbf,50.10701,8.66341,1,,
FFM_101,FFM Hbf,50.10739,8.66333,0,FFM,101
FFM_10,FFM Hbf,50.10593,8.66118,0,FFM,10
FFM_12,FFM Hbf,50.10658,8.66178,0,FFM,12
de:6412:10:6:1,FFM Hbf U-Bahn,50.107577,8.6638173,0,FFM,U4
LANGEN,Langen,49.99359,8.65677,1,,1
FFM_HAUPT,FFM Hauptwache,50.11403,8.67835,1,,
FFM_HAUPT_U,Hauptwache U1/U2/U3/U8,50.11385,8.67912,0,FFM_HAUPT,
FFM_HAUPT_S,FFM Hauptwache S,50.11404,8.67824,0,FFM_HAUPT,

# routes.txt
route_id,agency_id,route_short_name,route_long_name,route_desc,route_type
S3,DB,S3,,,109
U4,DB,U4,,,402
ICE,DB,ICE,,,101

# trips.txt
route_id,service_id,trip_id,trip_headsign,block_id
S3,S1,S3,,
U4,S1,U4,,
ICE,S1,ICE,,

# stop_times.txt
trip_id,arrival_time,departure_time,stop_id,stop_sequence,pickup_type,drop_off_type
S3,01:15:00,01:15:00,FFM_101,1,0,0
S3,01:20:00,01:20:00,FFM_HAUPT_S,2,0,0
U4,01:05:00,01:05:00,de:6412:10:6:1,0,0,0
U4,01:10:00,01:10:00,FFM_HAUPT_U,1,0,0
ICE,00:35:00,00:35:00,DA_10,0,0,0
ICE,00:45:00,00:45:00,FFM_10,1,0,0

# calendar_dates.txt
service_id,date,exception_type
S1,20190501,1

# frequencies.txt
trip_id,start_time,end_time,headway_secs
S3,01:15:00,25:15:00,3600
ICE,00:35:00,24:35:00,3600
U4,01:05:00,25:01:00,3600
)";

[[maybe_unused]] constexpr auto const kOjpGeocodingRequest =
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
          <siri:StopPointRef>test_DA_3</siri:StopPointRef>
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
      <JourneyRef>20190501_00:35_test_ICE</JourneyRef>
      <OperatingDayRef>2019-05-01</OperatingDayRef>
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
    <siri:RequestTimestamp>2019-05-01T00:30:00Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPTripRequest>
      <siri:RequestTimestamp>2019-05-01T00:30:00Z</siri:RequestTimestamp>
      <Origin>
        <PlaceRef>
          <StopPlaceRef>test_DA</StopPlaceRef>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
        <DepArrTime>2019-05-01T00:30:00Z</DepArrTime>
        <IndividualTransportOption />
      </Origin>
      <Destination>
        <PlaceRef>
          <StopPlaceRef>test_FFM</StopPlaceRef>
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
        <IncludeLegProjection>true</IncludeLegProjection>
        <IncludeTurnDescription>true</IncludeTurnDescription>
        <IncludeIntermediateStops>true</IncludeIntermediateStops>
        <UseRealtimeData>explanatory</UseRealtimeData>
      </Params>
    </OJPTripRequest>
  </siri:ServiceRequest>
</OJPRequest>
</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpIntermodalRoutingRequest =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns="http://www.vdv.de/ojp" xmlns:siri="http://www.siri.org.uk/siri" version="2.0">
<OJPRequest>
  <siri:ServiceRequest>
    <siri:ServiceRequestContext>
      <siri:Language>de</siri:Language>
    </siri:ServiceRequestContext>
    <siri:RequestTimestamp>2019-05-01T01:15Z</siri:RequestTimestamp>
    <siri:RequestorRef>OJP_DemoApp_Beta_OJP2.0</siri:RequestorRef>
    <OJPTripRequest>
      <siri:RequestTimestamp>2019-05-01T01:15Z</siri:RequestTimestamp>
      <Origin>
        <PlaceRef>
          <GeoPosition>
            <siri:Longitude>8.6586978</siri:Longitude>
            <siri:Latitude>50.1040763</siri:Latitude>
            <Properties />
          </GeoPosition>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
        <DepArrTime>2019-05-01T01:15Z</DepArrTime>
      </Origin>
      <Destination>
        <PlaceRef>
          <GeoPosition>
            <siri:Longitude>8.6767235</siri:Longitude>
            <siri:Latitude>50.1132737</siri:Latitude>
            <Properties />
          </GeoPosition>
          <Name>
            <Text>n/a</Text>
          </Name>
        </PlaceRef>
      </Destination>
      <Params>
        <NumberOfResults>5</NumberOfResults>
        <IncludeAllRestrictedLines>true</IncludeAllRestrictedLines>
        <IncludeTrackSections>true</IncludeTrackSections>
        <IncludeLegProjection>true</IncludeLegProjection>
        <IncludeIntermediateStops>true</IncludeIntermediateStops>
      </Params>
    </OJPTripRequest>
  </siri:ServiceRequest>
</OJPRequest>

</OJP>
)";

[[maybe_unused]] constexpr auto const kOjpGeocodingResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>MSG</siri:ResponseMessageIdentifier>
      <OJPLocationInformationDelivery>
        <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
      </OJPLocationInformationDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

[[maybe_unused]] constexpr auto const kOjpMapStopsResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>MSG</siri:ResponseMessageIdentifier>
      <OJPLocationInformationDelivery>
        <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_DA</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">DA Hbf</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">DA Hbf</Text>
            </Name>
            <GeoPosition>
              <siri:Longitude>8.63085</siri:Longitude>
              <siri:Latitude>49.8726</siri:Latitude>
            </GeoPosition>
            <Mode>
              <PtMode>rail</PtMode>
              <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
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
                <Text xml:lang="de">DA Hbf</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">DA Hbf</Text>
            </Name>
            <GeoPosition>
              <siri:Longitude>8.63003</siri:Longitude>
              <siri:Latitude>49.87355</siri:Latitude>
            </GeoPosition>
            <Mode>
              <PtMode>rail</PtMode>
              <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
            </Mode>
          </Place>
          <Complete>true</Complete>
          <Probability>1</Probability>
        </PlaceResult>
        <PlaceResult>
          <Place>
            <StopPlace>
              <StopPlaceRef>test_DA_10</StopPlaceRef>
              <StopPlaceName>
                <Text xml:lang="de">DA Hbf</Text>
              </StopPlaceName>
            </StopPlace>
            <Name>
              <Text xml:lang="de">DA Hbf</Text>
            </Name>
            <GeoPosition>
              <siri:Longitude>8.62926</siri:Longitude>
              <siri:Latitude>49.87336</siri:Latitude>
            </GeoPosition>
            <Mode>
              <PtMode>rail</PtMode>
              <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
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
      <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>MSG</siri:ResponseMessageIdentifier>
      <OJPStopEventDelivery>
        <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <StopEventResponseContext>
          <Places />
          <Situations />
        </StopEventResponseContext>
      </OJPStopEventDelivery>
    </siri:ServiceDelivery>
  </OJPResponse>
</OJP>)";

[[maybe_unused]] constexpr auto const kOjpTripInfoResponse =
    R"(<?xml version="1.0" encoding="utf-8"?>
<OJP xmlns:siri="http://www.siri.org.uk/siri" xmlns="http://www.vdv.de/ojp" version="2.0">
  <OJPResponse>
    <siri:ServiceDelivery>
      <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>MSG</siri:ResponseMessageIdentifier>
      <OJPTripInfoDelivery>
        <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <TripInfoResponseContext>
          <Places>
            <Place>
              <StopPlace>
                <siri:StopPlaceRef>test_DA</siri:StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">DA Hbf</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">DA Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.63085</siri:Longitude>
                <siri:Latitude>49.8726</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">DA Hbf</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">DA Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.62926</siri:Longitude>
                <siri:Latitude>49.87336</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPlace>
                <siri:StopPlaceRef>test_FFM</siri:StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">FFM Hbf</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">FFM Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.66341</siri:Longitude>
                <siri:Latitude>50.10701</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">FFM Hbf</Text>
                </StopPointName>
                <ParentRef>test_FFM</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">FFM Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.66118</siri:Longitude>
                <siri:Latitude>50.10593</siri:Latitude>
              </GeoPosition>
            </Place>
          </Places>
          <Situations />
        </TripInfoResponseContext>
        <TripInfoResult>
          <PreviousCall>
            <siri:StopPointRef>test_DA_10</siri:StopPointRef>
            <StopPointName>
              <Text xml:lang="de">DA Hbf</Text>
            </StopPointName>
            <PlannedQuay>
              <Text xml:lang="de">10</Text>
            </PlannedQuay>
            <NameSuffix>
              <Text xml:lang="de">PLATFORM_ACCESS_WITHOUT_ASSISTANCE</Text>
            </NameSuffix>
            <ServiceArrival />
            <ServiceDeparture>
              <TimetabledTime>2019-04-30T22:35:00Z</TimetabledTime>
              <EstimatedTime>2019-04-30T22:35:00Z</EstimatedTime>
            </ServiceDeparture>
            <Order>1</Order>
          </PreviousCall>
          <PreviousCall>
            <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
            <StopPointName>
              <Text xml:lang="de">FFM Hbf</Text>
            </StopPointName>
            <PlannedQuay>
              <Text xml:lang="de">10</Text>
            </PlannedQuay>
            <NameSuffix>
              <Text xml:lang="de">PLATFORM_ACCESS_WITHOUT_ASSISTANCE</Text>
            </NameSuffix>
            <ServiceArrival>
              <TimetabledTime>2019-04-30T22:45:00Z</TimetabledTime>
              <EstimatedTime>2019-04-30T22:45:00Z</EstimatedTime>
            </ServiceArrival>
            <ServiceDeparture />
            <Order>2</Order>
          </PreviousCall>
          <Service>
            <OperatingDayRef>2019-05-01</OperatingDayRef>
            <JourneyRef>20190501_00:35_test_ICE</JourneyRef>
            <PublicCode>ICE</PublicCode>
            <siri:LineRef>test_ICE</siri:LineRef>
            <siri:DirectionRef>0</siri:DirectionRef>
            <Mode>
              <PtMode>rail</PtMode>
              <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
            </Mode>
            <PublishedServiceName>
              <Text xml:lang="de">ICE</Text>
            </PublishedServiceName>
            <TrainNumber></TrainNumber>
            <OriginText>
              <Text xml:lang="de">DA Hbf</Text>
            </OriginText>
            <siri:OperatorRef>DB</siri:OperatorRef>
            <DestinationStopPointRef>test_FFM_10</DestinationStopPointRef>
            <DestinationText>
              <Text xml:lang="de">FFM Hbf</Text>
            </DestinationText>
          </Service>
          <JourneyTrack>
            <TrackSection>
              <TrackSectionStart>
                <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                <Name>
                  <Text xml:lang="de">DA Hbf</Text>
                </Name>
              </TrackSectionStart>
              <TrackSectionEnd>
                <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                <Name>
                  <Text xml:lang="de">FFM Hbf</Text>
                </Name>
              </TrackSectionEnd>
              <LinkProjection>
                <Position>
                  <siri:Longitude>8.62926</siri:Longitude>
                  <siri:Latitude>49.87336</siri:Latitude>
                </Position>
                <Position>
                  <siri:Longitude>8.66118</siri:Longitude>
                  <siri:Latitude>50.10593</siri:Latitude>
                </Position>
              </LinkProjection>
              <Duration>PT10M</Duration>
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
      <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
      <siri:ProducerRef>MOTIS</siri:ProducerRef>
      <siri:ResponseMessageIdentifier>MSG</siri:ResponseMessageIdentifier>
      <OJPTripDelivery>
        <siri:ResponseTimestamp>NOW</siri:ResponseTimestamp>
        <siri:DefaultLanguage>de</siri:DefaultLanguage>
        <TripResponseContext>
          <Places>
            <Place>
              <StopPlace>
                <siri:StopPlaceRef>test_DA</siri:StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">DA Hbf</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">DA Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.63085</siri:Longitude>
                <siri:Latitude>49.8726</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">DA Hbf</Text>
                </StopPointName>
                <ParentRef>test_DA</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">DA Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.62926</siri:Longitude>
                <siri:Latitude>49.87336</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPlace>
                <siri:StopPlaceRef>test_FFM</siri:StopPlaceRef>
                <StopPlaceName>
                  <Text xml:lang="de">FFM Hbf</Text>
                </StopPlaceName>
              </StopPlace>
              <Name>
                <Text xml:lang="de">FFM Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.66341</siri:Longitude>
                <siri:Latitude>50.10701</siri:Latitude>
              </GeoPosition>
            </Place>
            <Place>
              <StopPoint>
                <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                <StopPointName>
                  <Text xml:lang="de">FFM Hbf</Text>
                </StopPointName>
                <ParentRef>test_FFM</ParentRef>
              </StopPoint>
              <Name>
                <Text xml:lang="de">FFM Hbf</Text>
              </Name>
              <GeoPosition>
                <siri:Longitude>8.66118</siri:Longitude>
                <siri:Latitude>50.10593</siri:Latitude>
              </GeoPosition>
            </Place>
          </Places>
        </TripResponseContext>
        <TripResult>
          <Id>1</Id>
          <Trip>
            <Id>1</Id>
            <Duration>PT10M</Duration>
            <StartTime>2019-05-01T00:35:00Z</StartTime>
            <EndTime>2019-05-01T00:45:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>2</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT10M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">DA Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2019-05-01T00:35:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T00:35:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2019-05-01T00:45:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T00:45:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>20190501</OperatingDayRef>
                  <JourneyRef>20190501_02:35_test_ICE</JourneyRef>
                  <LineRef>test_ICE</LineRef>
                  <DirectionRef>0</DirectionRef>
                  <siri:OperatorRef>DB</siri:OperatorRef>
                  <ProductCategory />
                  <DestinationText>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </DestinationText>
                  <PublishedServiceName>
                    <Text xml:lang="de">ICE</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>rail</PtMode>
                    <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
                  </Mode>
                </Service>
                <LegTrack>
                  <TrackSection>
                    <TrackSectionStart>
                      <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                      <Name>
                        <Text xml:lang="de">DA Hbf</Text>
                      </Name>
                      <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                    </TrackSectionStart>
                    <TrackSectionEnd>
                      <Name>
                        <Text xml:lang="de">FFM Hbf</Text>
                      </Name>
                    </TrackSectionEnd>
                    <LinkProjection>
                      <Position>
                        <siri:Longitude>8.62926</siri:Longitude>
                        <siri:Latitude>49.87336</siri:Latitude>
                      </Position>
                      <Position>
                        <siri:Longitude>8.66118</siri:Longitude>
                        <siri:Latitude>50.10593</siri:Latitude>
                      </Position>
                    </LinkProjection>
                    <Duration>PT10M</Duration>
                    <Length>2</Length>
                  </TrackSection>
                </LegTrack>
              </TimedLeg>
            </Leg>
          </Trip>
        </TripResult>
        <TripResult>
          <Id>2</Id>
          <Trip>
            <Id>2</Id>
            <Duration>PT10M</Duration>
            <StartTime>2019-05-01T01:35:00Z</StartTime>
            <EndTime>2019-05-01T01:45:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>2</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT10M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">DA Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2019-05-01T01:35:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T01:35:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2019-05-01T01:45:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T01:45:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>20190501</OperatingDayRef>
                  <JourneyRef>20190501_03:35_test_ICE</JourneyRef>
                  <LineRef>test_ICE</LineRef>
                  <DirectionRef>0</DirectionRef>
                  <siri:OperatorRef>DB</siri:OperatorRef>
                  <ProductCategory />
                  <DestinationText>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </DestinationText>
                  <PublishedServiceName>
                    <Text xml:lang="de">ICE</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>rail</PtMode>
                    <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
                  </Mode>
                </Service>
                <LegTrack>
                  <TrackSection>
                    <TrackSectionStart>
                      <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                      <Name>
                        <Text xml:lang="de">DA Hbf</Text>
                      </Name>
                      <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                    </TrackSectionStart>
                    <TrackSectionEnd>
                      <Name>
                        <Text xml:lang="de">FFM Hbf</Text>
                      </Name>
                    </TrackSectionEnd>
                    <LinkProjection>
                      <Position>
                        <siri:Longitude>8.62926</siri:Longitude>
                        <siri:Latitude>49.87336</siri:Latitude>
                      </Position>
                      <Position>
                        <siri:Longitude>8.66118</siri:Longitude>
                        <siri:Latitude>50.10593</siri:Latitude>
                      </Position>
                    </LinkProjection>
                    <Duration>PT10M</Duration>
                    <Length>2</Length>
                  </TrackSection>
                </LegTrack>
              </TimedLeg>
            </Leg>
          </Trip>
        </TripResult>
        <TripResult>
          <Id>3</Id>
          <Trip>
            <Id>3</Id>
            <Duration>PT10M</Duration>
            <StartTime>2019-05-01T02:35:00Z</StartTime>
            <EndTime>2019-05-01T02:45:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>2</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT10M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">DA Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2019-05-01T02:35:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T02:35:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2019-05-01T02:45:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T02:45:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>20190501</OperatingDayRef>
                  <JourneyRef>20190501_04:35_test_ICE</JourneyRef>
                  <LineRef>test_ICE</LineRef>
                  <DirectionRef>0</DirectionRef>
                  <siri:OperatorRef>DB</siri:OperatorRef>
                  <ProductCategory />
                  <DestinationText>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </DestinationText>
                  <PublishedServiceName>
                    <Text xml:lang="de">ICE</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>rail</PtMode>
                    <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
                  </Mode>
                </Service>
                <LegTrack>
                  <TrackSection>
                    <TrackSectionStart>
                      <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                      <Name>
                        <Text xml:lang="de">DA Hbf</Text>
                      </Name>
                      <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                    </TrackSectionStart>
                    <TrackSectionEnd>
                      <Name>
                        <Text xml:lang="de">FFM Hbf</Text>
                      </Name>
                    </TrackSectionEnd>
                    <LinkProjection>
                      <Position>
                        <siri:Longitude>8.62926</siri:Longitude>
                        <siri:Latitude>49.87336</siri:Latitude>
                      </Position>
                      <Position>
                        <siri:Longitude>8.66118</siri:Longitude>
                        <siri:Latitude>50.10593</siri:Latitude>
                      </Position>
                    </LinkProjection>
                    <Duration>PT10M</Duration>
                    <Length>2</Length>
                  </TrackSection>
                </LegTrack>
              </TimedLeg>
            </Leg>
          </Trip>
        </TripResult>
        <TripResult>
          <Id>4</Id>
          <Trip>
            <Id>4</Id>
            <Duration>PT10M</Duration>
            <StartTime>2019-05-01T03:35:00Z</StartTime>
            <EndTime>2019-05-01T03:45:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>2</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT10M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">DA Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2019-05-01T03:35:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T03:35:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2019-05-01T03:45:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T03:45:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>20190501</OperatingDayRef>
                  <JourneyRef>20190501_05:35_test_ICE</JourneyRef>
                  <LineRef>test_ICE</LineRef>
                  <DirectionRef>0</DirectionRef>
                  <siri:OperatorRef>DB</siri:OperatorRef>
                  <ProductCategory />
                  <DestinationText>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </DestinationText>
                  <PublishedServiceName>
                    <Text xml:lang="de">ICE</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>rail</PtMode>
                    <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
                  </Mode>
                </Service>
                <LegTrack>
                  <TrackSection>
                    <TrackSectionStart>
                      <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                      <Name>
                        <Text xml:lang="de">DA Hbf</Text>
                      </Name>
                      <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                    </TrackSectionStart>
                    <TrackSectionEnd>
                      <Name>
                        <Text xml:lang="de">FFM Hbf</Text>
                      </Name>
                    </TrackSectionEnd>
                    <LinkProjection>
                      <Position>
                        <siri:Longitude>8.62926</siri:Longitude>
                        <siri:Latitude>49.87336</siri:Latitude>
                      </Position>
                      <Position>
                        <siri:Longitude>8.66118</siri:Longitude>
                        <siri:Latitude>50.10593</siri:Latitude>
                      </Position>
                    </LinkProjection>
                    <Duration>PT10M</Duration>
                    <Length>2</Length>
                  </TrackSection>
                </LegTrack>
              </TimedLeg>
            </Leg>
          </Trip>
        </TripResult>
        <TripResult>
          <Id>5</Id>
          <Trip>
            <Id>5</Id>
            <Duration>PT10M</Duration>
            <StartTime>2019-05-01T04:35:00Z</StartTime>
            <EndTime>2019-05-01T04:45:00Z</EndTime>
            <Transfers>0</Transfers>
            <Distance>2</Distance>
            <Leg>
              <Id>1</Id>
              <Duration>PT10M</Duration>
              <TimedLeg>
                <LegBoard>
                  <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">DA Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceDeparture>
                    <TimetabledTime>2019-05-01T04:35:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T04:35:00Z</EstimatedTime>
                  </ServiceDeparture>
                  <Order>1</Order>
                </LegBoard>
                <LegAlight>
                  <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                  <StopPointName>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </StopPointName>
                  <PlannedQuay>
                    <Text xml:lang="de">10</Text>
                  </PlannedQuay>
                  <ServiceArrival>
                    <TimetabledTime>2019-05-01T04:45:00Z</TimetabledTime>
                    <EstimatedTime>2019-05-01T04:45:00Z</EstimatedTime>
                  </ServiceArrival>
                  <Order>2</Order>
                </LegAlight>
                <Service>
                  <OperatingDayRef>20190501</OperatingDayRef>
                  <JourneyRef>20190501_06:35_test_ICE</JourneyRef>
                  <LineRef>test_ICE</LineRef>
                  <DirectionRef>0</DirectionRef>
                  <siri:OperatorRef>DB</siri:OperatorRef>
                  <ProductCategory />
                  <DestinationText>
                    <Text xml:lang="de">FFM Hbf</Text>
                  </DestinationText>
                  <PublishedServiceName>
                    <Text xml:lang="de">ICE</Text>
                  </PublishedServiceName>
                  <Mode>
                    <PtMode>rail</PtMode>
                    <siri:RailSubmode>highSpeedRail</siri:RailSubmode>
                  </Mode>
                </Service>
                <LegTrack>
                  <TrackSection>
                    <TrackSectionStart>
                      <siri:StopPointRef>test_DA_10</siri:StopPointRef>
                      <Name>
                        <Text xml:lang="de">DA Hbf</Text>
                      </Name>
                      <siri:StopPointRef>test_FFM_10</siri:StopPointRef>
                    </TrackSectionStart>
                    <TrackSectionEnd>
                      <Name>
                        <Text xml:lang="de">FFM Hbf</Text>
                      </Name>
                    </TrackSectionEnd>
                    <LinkProjection>
                      <Position>
                        <siri:Longitude>8.62926</siri:Longitude>
                        <siri:Latitude>49.87336</siri:Latitude>
                      </Position>
                      <Position>
                        <siri:Longitude>8.66118</siri:Longitude>
                        <siri:Latitude>50.10593</siri:Latitude>
                      </Position>
                    </LinkProjection>
                    <Duration>PT10M</Duration>
                    <Length>2</Length>
                  </TrackSection>
                </LegTrack>
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

  auto const c =
      config{.osm_ = {"test/resources/test_case.osm.pbf"},
             .timetable_ =
                 config::timetable{.first_day_ = "2019-05-01",
                                   .num_days_ = 2,
                                   .datasets_ = {{"test", {.path_ = kGTFS}}}},
             .street_routing_ = true,
             .osr_footpath_ = true,
             .geocoding_ = true};
  import(c, "test/data", true);
  auto d = data{"test/data", c};
  d.init_rtt(date::sys_days{2019_y / May / 1});

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

  auto const normalize_response = [](std::string_view input) {
    auto out = std::string{input};

    auto const normalize_tag = [&](std::string_view start_tag,
                                   std::string_view end_tag,
                                   std::string_view replacement) {
      auto pos = std::size_t{0};
      while ((pos = out.find(start_tag, pos)) != std::string::npos) {
        auto const value_start = pos + start_tag.size();
        auto const value_end = out.find(end_tag, value_start);
        if (value_end == std::string::npos) {
          break;
        }
        out.replace(value_start, value_end - value_start, replacement);
        pos = value_end + end_tag.size();
      }
    };

    normalize_tag("<siri:ResponseTimestamp>", "</siri:ResponseTimestamp>",
                  "NOW");
    normalize_tag("<siri:ResponseMessageIdentifier>",
                  "</siri:ResponseMessageIdentifier>", "MSG");

    return out;
  };

  auto const expect_response = [&](net::reply const& reply,
                                   std::string_view expected) {
    auto const* res = std::get_if<net::web_server::string_res_t>(&reply);
    ASSERT_NE(nullptr, res);
    EXPECT_EQ(boost::beast::http::status::ok, res->result());
    EXPECT_EQ("text/xml; charset=utf-8",
              res->base()[boost::beast::http::field::content_type]);
    EXPECT_EQ(normalize_response(expected), normalize_response(res->body()));
  };

  expect_response(send_request(kOjpGeocodingRequest), kOjpGeocodingResponse);
  expect_response(send_request(kOjpMapStopsRequest), kOjpMapStopsResponse);
  expect_response(send_request(kOjpStopEventRequest), kOjpStopEventResponse);
  expect_response(send_request(kOjpTripInfoRequest), kOjpTripInfoResponse);
  expect_response(send_request(kOjpRoutingRequest), kOjpRoutingResponse);
}
