#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

namespace motis::ris::risml {

constexpr auto const connection_assessment_fixture_1 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007051838050" KNr="123984009">
<ListNachricht><Nachricht>
 <Anschlussbewertung>
  <ZE Typ="An">
    <Bf EvaNr="8001585"/>
    <ListAnschl>
      <Anschl Bewertung="2">
        <ZE Typ="Ab">
          <Bf EvaNr="8001585"/>
          <Service Id="85751154" IdBfEvaNr="8004005" IdTyp="Ab"
                   IdVerwaltung="M2" IdZGattung="S" IdZGattungInt="DPN"
                   IdZNr="90708" IdZeit="20151007052500" SourceZNR="EFZ"
                   ZielBfEvaNr="8000430" Zielzeit="20151007061600" IdLinie="42">
            <ListZug/>
          </Service>
          <Zeit Ist="" Prog="" Soll="20151007055000"/>
          <Zug Gattung="S" GattungInt="DPN" Nr="90708" Verwaltung="M2" Linie="42">
            <ListZE/>
          </Zug>
        </ZE>
      </Anschl>
    </ListAnschl>
    <Service Id="86090468" IdBfEvaNr="8000253" IdTyp="An" IdVerwaltung="03"
             IdZGattung="S" IdZGattungInt="s" IdZNr="30815" IdZeit="20151007051400"
             SourceZNR="EFZ" ZielBfEvaNr="8000142" Zielzeit="20151007065800"
             IdLinie="23">
      <ListZug/>
    </Service>
    <Zeit Ist="" Prog="" Soll="20151007054400"/>
    <Zug Gattung="S" GattungInt="s" Nr="30815" Verwaltung="03" Linie="23">
      <ListZE/>
    </Zug>
  </ZE>
 </Anschlussbewertung>
 <ListQuelle>
 <Quelle Sender="RSL" Typ="Anschlussbewertung" KNr="15100705182500000"
         TIn="20151007051838039" TOutSnd="20151007051825"/>
 </ListQuelle>
</Nachricht></ListNachricht>
</Paket>)";

TEST(DISABLED_ris_connection_assessment_message, message_1) {
  auto const messages = parse(connection_assessment_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444187918, message.timestamp_);
  EXPECT_EQ(1444187640, message.earliest_);
  EXPECT_EQ(1444193880, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_ConnectionAssessmentMessage,
            outer_msg->content_type());
  auto inner_msg = reinterpret_cast<ConnectionAssessmentMessage const*>(
      outer_msg->content());

  auto from_id = inner_msg->from_trip_id();
  EXPECT_STREQ("8000253", from_id->station_id()->c_str());
  EXPECT_EQ(30815, from_id->service_num());
  EXPECT_EQ(1444187640, from_id->schedule_time());

  auto from = inner_msg->from();
  EXPECT_STREQ("8001585", from->station_id()->c_str());
  EXPECT_EQ(30815, from->service_num());
  EXPECT_STREQ("23", from->line_id()->c_str());
  EXPECT_EQ(1444189440, from->schedule_time());
  EXPECT_EQ(EventType_ARR, from->type());

  auto to = inner_msg->to();
  ASSERT_EQ(1, to->size());

  auto e0 = to->Get(0);
  EXPECT_STREQ("8001585", e0->base()->station_id()->c_str());
  EXPECT_EQ(90708, e0->base()->service_num());
  EXPECT_STREQ("42", e0->base()->line_id()->c_str());
  EXPECT_EQ(1444189800, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());

  EXPECT_EQ(2, e0->assessment());

  auto e0_id = e0->trip_id();
  EXPECT_STREQ("8004005", e0_id->station_id()->c_str());
  EXPECT_EQ(90708, e0_id->service_num());
  EXPECT_EQ(1444188300, e0_id->schedule_time());
}

constexpr auto const connection_assessment_fixture_2 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151006235948056" KNr="123863851">
 <ListNachricht>
  <Nachricht>
   <Anschlussbewertung>
    <ZE Typ="Ziel">
     <Bf EvaNr="8000261"/>
     <ListAnschl>
      <Anschl Bewertung="4">
       <ZE Typ="Ab">
        <Bf EvaNr="8098263"/>
        <Service Id="85777037" IdBfEvaNr="8002980" IdTyp="Ab" IdVerwaltung="07"
                 IdZGattung="S" IdZGattungInt="s" IdZNr="8326" IdZeit="20151006233600"
                 SourceZNR="EFZ" ZielBfEvaNr="8004204" Zielzeit="20151007010500">
         <ListZug/>
        </Service>
        <Zeit Ist="" Prog="" Soll="20151007001900"/>
        <Zug Gattung="S" GattungInt="s" Nr="8326" Verwaltung="07">
         <ListZE/>
        </Zug>
       </ZE>
      </Anschl>
      <Anschl Bewertung="3">
       <ZE Typ="Ab">
        <Bf EvaNr="8098263"/>
        <Service Id="85967814" IdBfEvaNr="8002347" IdTyp="Ab" IdVerwaltung="07"
                 IdZGattung="S" IdZGattungInt="s" IdZNr="8426" IdZeit="20151006234100"
                 SourceZNR="EFZ" ZielBfEvaNr="8000119" Zielzeit="20151007005600">
         <ListZug/>
        </Service>
        <Zeit Ist="" Prog="" Soll="20151007002100"/>
        <Zug Gattung="S" GattungInt="s" Nr="8426" Verwaltung="07">
         <ListZE/>
        </Zug>
       </ZE>
      </Anschl>
     </ListAnschl>
     <Service Id="86059254" IdBfEvaNr="8004775" IdTyp="Ziel" IdVerwaltung="07"
              IdZGattung="S" IdZGattungInt="s" IdZNr="8239" IdZeit="20151006233200"
              SourceZNR="EFZ" ZielBfEvaNr="8000261" Zielzeit="20151007000800">
      <ListZug/>
     </Service>
     <Zeit Ist="" Prog="" Soll="20151007000800"/>
     <Zug Gattung="S" GattungInt="s" Nr="8239" Verwaltung="07">
      <ListZE/>
     </Zug>
    </ZE>
   </Anschlussbewertung>
   <ListQuelle>
    <Quelle Sender="RSL" Typ="Anschlussbewertung" KNr="15100623594700000"
     TIn="20151006235944623" TOutSnd="20151006235947"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(DISABLED_ris_connection_assessment_message, message_2) {
  auto const messages = parse(connection_assessment_fixture_2);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168788, message.timestamp_);
  EXPECT_EQ(1444167120, message.earliest_);
  EXPECT_EQ(1444172700, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_ConnectionAssessmentMessage,
            outer_msg->content_type());
  auto inner_msg = reinterpret_cast<ConnectionAssessmentMessage const*>(
      outer_msg->content());

  auto from = inner_msg->from();
  EXPECT_STREQ("8000261", from->station_id()->c_str());
  EXPECT_EQ(8239, from->service_num());
  EXPECT_STREQ("", from->line_id()->c_str());
  EXPECT_EQ(1444169280, from->schedule_time());
  EXPECT_EQ(EventType_ARR, from->type());

  auto to = inner_msg->to();
  ASSERT_EQ(2, to->size());

  auto e0 = to->Get(0);
  EXPECT_STREQ("8098263", e0->base()->station_id()->c_str());
  EXPECT_EQ(8326, e0->base()->service_num());
  EXPECT_STREQ("", e0->base()->line_id()->c_str());
  EXPECT_EQ(1444169940, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());

  EXPECT_EQ(4, e0->assessment());

  auto e1 = to->Get(1);
  EXPECT_STREQ("8098263", e1->base()->station_id()->c_str());
  EXPECT_EQ(8426, e1->base()->service_num());
  EXPECT_STREQ("", e1->base()->line_id()->c_str());
  EXPECT_EQ(1444170060, e1->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->base()->type());

  EXPECT_EQ(3, e1->assessment());
}

}  // namespace motis::ris::risml
