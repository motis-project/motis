#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

using namespace motis;
using namespace motis::ris;
using namespace motis::ris::risml;

auto const free_text_fixture_1 = R"(<?xml version="1.0"?>
<Paket Version="1.2" SpezVer="1" TOut="20151007070635992" KNr="124264761">
 <ListNachricht>
  <Nachricht>
   <Freitext>
    <FT Typ="Intern" Code="36" Text="technische Störung"/>
    <ListService>
     <Service Id="85721079" IdZNr="99655" IdZGattung="Bus"
              IdBf="CCTVI" IdBfEvaNr="0460711" IdZeit="20151007073000"
              ZielBfCode="CBESC" ZielBfEvaNr="0683407" Zielzeit="20151007075700"
              IdVerwaltung="ovfNBG" IdZGattungInt="Bus" IdLinie="818"
              SourceZNr="EFZ">
      <ListZug>
       <Zug Nr="99655" Gattung="Bus" Linie="818"
            GattungInt="Bus" Verwaltung="ovfNBG">
        <ListZE>
         <ZE Typ="An">
          <Bf Code="CDMJP" EvaNr="0680414" Name="Bensenstr., Rothenburg"/>
          <Zeit Soll="20151007075400"/>
          <Gleis Soll="2"/>
         </ZE>
         <ZE Typ="Ab">
          <Bf Code="CDMJP" EvaNr="0680414" Name="Bensenstr., Rothenburg"/>
          <Zeit Soll="20151007075400"/>
          <Gleis Soll="1"/>
         </ZE>
        </ListZE>
       </Zug>
      </ListZug>
     </Service>
    </ListService>
   </Freitext>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

constexpr auto const free_text_fixture_2 = R"(<?xml version="1.0"?>
<Paket Version="1.2" SpezVer="1" TOut="20160410114622030" KNr="1229864054">
 <ListNachricht>
  <Nachricht>
   <Freitext>
    <FT Typ="Intern" Text="Wartet auf IC 2371 in Hannover Hbf."
        Erzeugt="20160410114620" Loeschen="20160411114620"/>
    <ListService>
     <Service Id="317852902598" IdZGattung="WFB" IdZGattungInt="DPN" IdBf="HR"
              IdBfEvaNr="8000316" IdZeit="20160410103800" IdZNr="75779"
              ZielBfCode="HBS" ZielBfEvaNr="8000049" Zielzeit="20160410134100"
              IdVerwaltung="W3" SourceZNr="EFZ" IdLinie="RE60">
      <ListZug>
       <Zug Nr="75779" Gattung="WFB" GattungInt="DPN" Linie="RE60"
            Name="WFB 75779" Verwaltung="W3">
        <ListZE>
         <ZE Typ="An">
          <Bf Code="HH" EvaNr="8000152" Name="Hannover Hbf"/>
          <Zeit Soll="20160410125100"/>
         </ZE>
         <ZE Typ="Ab">
          <Bf Code="HH" EvaNr="8000152" Name="Hannover Hbf"/>
          <Zeit Soll="20160410125500"/>
         </ZE>
        </ListZE>
       </Zug>
      </ListZug>
     </Service>
    </ListService>
    <ListBf/>
   </Freitext>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_free_text_message, free_text_test) {
  auto const messages = parse(free_text_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444194395, message.timestamp_);
  EXPECT_EQ(1444195800, message.earliest_);
  EXPECT_EQ(1444197420, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_FreeTextMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<FreeTextMessage const*>(outer_msg->content());

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

  auto free_text = inner_msg->free_text();
  EXPECT_EQ(36, free_text->code());
  EXPECT_EQ("technische Störung", free_text->text()->str());
  EXPECT_EQ("Intern", free_text->type()->str());
}

TEST(ris_free_text_message, free_text_test2) {
  auto const messages = parse(free_text_fixture_2);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1460281582, message.timestamp_);
  EXPECT_EQ(1460277480, message.earliest_);
  EXPECT_EQ(1460288460, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_FreeTextMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<FreeTextMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8000316", id->station_id()->c_str());
  EXPECT_EQ(75779, id->service_num());
  EXPECT_EQ(1460277480, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(2, events->size());

  auto e0 = events->Get(0);
  EXPECT_EQ(75779, e0->service_num());
  EXPECT_STREQ("8000152", e0->station_id()->c_str());
  EXPECT_EQ(1460285460, e0->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->type());

  auto e1 = events->Get(1);
  EXPECT_EQ(75779, e1->service_num());
  EXPECT_STREQ("8000152", e1->station_id()->c_str());
  EXPECT_EQ(1460285700, e1->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->type());

  auto free_text = inner_msg->free_text();
  EXPECT_EQ(0, free_text->code());
  EXPECT_EQ("Wartet auf IC 2371 in Hannover Hbf.", free_text->text()->str());
  EXPECT_EQ("Intern", free_text->type()->str());
}
