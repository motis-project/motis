#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

namespace motis::ris::risml {

auto const track_fixture_1 = R"(<?xml version="1.0"?>
<Paket Version="1.2" SpezVer="1" TOut="20151007070635992" KNr="124264761">
 <ListNachricht>
  <Nachricht>
   <Gleisaenderung>
    <Service Id="85721079" IdZNr="99655" IdZGattung="Bus" IdBf="CCTVI"
             IdBfEvaNr="0460711" IdZeit="20151007073000" ZielBfCode="CBESC"
             ZielBfEvaNr="0683407" Zielzeit="20151007075700"
             IdVerwaltung="ovfNBG" IdZGattungInt="Bus" IdLinie="818"
             SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="99655" Gattung="Bus" Linie="818"
           GattungInt="Bus" Verwaltung="ovfNBG">
       <ListZE>
        <ZE Typ="An">
         <Bf Code="CDMJP" EvaNr="0680414" Name="Bensenstr., Rothenburg"/>
         <Zeit Soll="20151007075400"/>
         <Gleis Soll="2" Prog="6"/>
        </ZE>
        <ZE Typ="Ab">
         <Bf Code="CDMJP" EvaNr="0680414" Name="Bensenstr., Rothenburg"/>
         <Zeit Soll="20151007075400"/>
         <Gleis Soll="1" Prog="5"/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Gleisaenderung>
  </Nachricht>
 </ListNachricht>
</Paket>)";

TEST(ris_track_message, track_test) {
  auto const messages = parse(track_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444194395, message.timestamp_);
  EXPECT_EQ(1444195800, message.earliest_);
  EXPECT_EQ(1444197420, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_TrackMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<TrackMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("0460711", id->station_id()->c_str());
  EXPECT_EQ(99655U, id->service_num());
  EXPECT_EQ(1444195800, id->schedule_time());

  auto events = inner_msg->events();
  ASSERT_EQ(2, events->size());

  auto e0 = events->Get(0);
  EXPECT_EQ(99655U, e0->base()->service_num());
  EXPECT_STREQ("0680414", e0->base()->station_id()->c_str());
  EXPECT_EQ(1444197240, e0->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->base()->type());
  EXPECT_EQ("6", e0->updated_track()->str());

  auto e1 = events->Get(1);
  EXPECT_EQ(99655U, e1->base()->service_num());
  EXPECT_STREQ("0680414", e1->base()->station_id()->c_str());
  EXPECT_EQ(1444197240, e1->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->base()->type());
  EXPECT_EQ("5", e1->updated_track()->str());
}

}  // namespace motis::ris::risml
