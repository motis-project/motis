#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

namespace motis::ris::risml {

constexpr auto const cancel_fixture_1 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007070635992" KNr="124264761">
 <ListNachricht>
  <Nachricht>
   <Ausfall>
    <Service Id="85721079" IdZNr="99655" IdZGattung="Bus" IdBf="CCTVI"
             IdBfEvaNr="0460711" IdZeit="20151007073000" ZielBfCode="CBESC"
             ZielBfEvaNr="0683407" Zielzeit="20151007075700"
             IdVerwaltung="ovfNBG" IdZGattungInt="Bus" IdLinie="818"
             SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="99655" Gattung="Bus" Linie="818" GattungInt="Bus"
           Verwaltung="ovfNBG">
       <ListZE>
        <ZE Typ="An" Status="Ausf">
         <Bf Code="CDMJP" EvaNr="0680414"
             Name="Bensenstr., Rothenburg ob der Taube"/>
         <Zeit Soll="20151007075400"/>
         <Gleis Soll=""/>
        </ZE>
        <ZE Typ="Ab" Status="Ausf">
         <Bf Code="CDMJP" EvaNr="0680414"
             Name="Bensenstr., Rothenburg ob der Taube"/>
         <Zeit Soll="20151007075400"/>
         <Gleis Soll=""/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ausfall>
   <ListQuelle>
    <Quelle Sender="RSL" Typ="Ausfall" KNr="64205100707004800093"
            TIn="20151007070635948" TOutSnd="20151007070050"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_cancel_message, message_1) {
  auto const messages = parse(cancel_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444194395, message.timestamp_);
  EXPECT_EQ(1444195800, message.earliest_);
  EXPECT_EQ(1444197420, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_CancelMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<CancelMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("0460711", id->station_id()->c_str());
  EXPECT_EQ(99655, id->service_num());
  EXPECT_EQ(1444195800, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(2, events->size());

  auto e0 = events->Get(0);
  EXPECT_EQ(99655, e0->service_num());
  EXPECT_STREQ("0680414", e0->station_id()->c_str());
  EXPECT_EQ(1444197240, e0->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->type());

  auto e1 = events->Get(1);
  EXPECT_EQ(99655, e1->service_num());
  EXPECT_STREQ("0680414", e1->station_id()->c_str());
  EXPECT_EQ(1444197240, e1->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->type());
}

constexpr auto const cancel_fixture_2 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007161500382" KNr="125683842">
 <ListNachricht>
  <Nachricht>
   <Ausfall>
    <Service Id="86318167" IdZNr="31126" IdZGattung="Bus" IdBf="CCJBN"
             IdBfEvaNr="0732798" IdZeit="20151007155500" ZielBfCode="CCHOU"
             ZielBfEvaNr="0730993" Zielzeit="20151007163600"
             IdVerwaltung="vbbBVB" IdZGattungInt="Bus" IdLinie="M21"
             SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="31126" Gattung="Bus" Linie="M21" GattungInt="Bus" Verwaltung="vbbBVB">
       <ListZE>
        <ZE Typ="An" Status="Ausf">
         <Bf Code="CCHOM" EvaNr="0730985" Name="Jungfernheide Bhf (S+U), Berlin"/>
         <Zeit Soll="20151007163500"/>
         <Gleis Soll=""/>
        </ZE>
        <ZE Typ="Ab" Status="Ausf">
         <Bf Code="CCHOM" EvaNr="0730985" Name="Jungfernheide Bhf (S+U), Berlin"/>
         <Zeit Soll="20151007163500"/>
         <Gleis Soll=""/>
        </ZE>
        <ZE Typ="Ziel" Status="Ausf">
         <Bf Code="CCHOU" EvaNr="0730993" Name="Goerdelersteg, Berlin"/>
         <Zeit Soll="20151007163600"/>
         <Gleis Soll=""/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ausfall>
   <ListQuelle>
    <Quelle Sender="RSL" Typ="Ausfall" KNr="11805100716140800086"
            TIn="20151007161500043" TOutSnd="20151007161409"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_ausfall_message, message_2) {
  auto const messages = parse(cancel_fixture_2);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444227300, message.timestamp_);
  EXPECT_EQ(1444226100, message.earliest_);
  EXPECT_EQ(1444228560, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_CancelMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<CancelMessage const*>(outer_msg->content());

  auto events = inner_msg->events();
  ASSERT_EQ(3, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("0730985", e0->station_id()->c_str());
  EXPECT_EQ(31126, e0->service_num());
  EXPECT_STREQ("M21", e0->line_id()->c_str());
  EXPECT_EQ(1444228500, e0->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->type());

  auto e1 = events->Get(1);
  EXPECT_STREQ("0730985", e1->station_id()->c_str());
  EXPECT_EQ(31126, e1->service_num());
  EXPECT_STREQ("M21", e1->line_id()->c_str());
  EXPECT_EQ(1444228500, e1->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->type());

  auto e2 = events->Get(2);
  EXPECT_STREQ("0730993", e2->station_id()->c_str());
  EXPECT_EQ(31126, e2->service_num());
  EXPECT_STREQ("M21", e2->line_id()->c_str());
  EXPECT_EQ(1444228560, e2->schedule_time());
  EXPECT_EQ(EventType_ARR, e2->type());
}

}  // namespace motis::ris::risml
