#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

namespace motis::ris::risml {

constexpr auto const reroute_fixture_1 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007000114447" KNr="123864543">
 <ListNachricht>
  <Nachricht>
   <Umleitung>
    <Service Id="85929022" IdZNr="2318" IdZGattung="IC" IdBf="RH"
             IdBfEvaNr="8000156" IdZeit="20151008182500" ZielBfCode="EDO"
             ZielBfEvaNr="8000080" Zielzeit="20151008222100" IdVerwaltung="80"
             IdZGattungInt="IC" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="2318" Gattung="IC" GattungInt="IC" Verwaltung="80">
       <ListZE>
        <ZE Typ="An" Status="UmlAusf">
         <Bf Code="EE" EvaNr="8000098" Name="Essen Hbf"/>
         <Zeit Soll="20151008215700"/>
        </ZE>
        <ZE Typ="Ab" Status="UmlAusf">
         <Bf Code="EE" EvaNr="8000098" Name="Essen Hbf"/>
         <Zeit Soll="20151008215900"/>
        </ZE>
        <ZE Typ="An" Status="UmlAusf">
         <Bf Code="EBO" EvaNr="8000041" Name="Bochum Hbf"/>
         <Zeit Soll="20151008220800"/>
        </ZE>
        <ZE Typ="Ab" Status="UmlAusf">
         <Bf Code="EBO" EvaNr="8000041" Name="Bochum Hbf"/>
         <Zeit Soll="20151008221000"/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
     <ListUml>
      <Uml>
       <ListZug>
        <Zug Nr="2318" Gattung="IC" GattungInt="IC" Verwaltung="80">
         <ListZE>
          <ZE Typ="An" RegSta="UmlNeu">
           <Bf Code="EG" EvaNr="8000118" Name="Gelsenkirchen Hbf"/>
           <Zeit Soll="20151008215900"/>
           <Gleis Soll="6"/>
          </ZE>
          <ZE Typ="Ab" RegSta="UmlNeu">
           <Bf Code="EG" EvaNr="8000118" Name="Gelsenkirchen Hbf"/>
           <Zeit Soll="20151008220100"/>
           <Gleis Soll="6"/>
          </ZE>
         </ListZE>
        </Zug>
       </ListZug>
      </Uml>
     </ListUml>
    </Service>
   </Umleitung>
   <ListQuelle>
    <Quelle Sender="ZENTRAL" Typ="Umleitung" KNr="18817" TIn="20151007000105223"
            TOutSnd="20151007000113597"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_reroute_message, message_1) {
  auto const messages = parse(reroute_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168874, message.timestamp_);
  EXPECT_EQ(1444321500, message.earliest_);
  EXPECT_EQ(1444335660, message.latest_);

  auto outer_msg = GetMessage(message.data());
  ASSERT_EQ(MessageUnion_RerouteMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<RerouteMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8000156", id->station_id()->c_str());
  EXPECT_EQ(2318, id->service_num());
  EXPECT_EQ(1444321500, id->schedule_time());

  auto cancelled_events = inner_msg->cancelled_events();
  ASSERT_EQ(4, cancelled_events->size());

  auto ce0 = cancelled_events->Get(0);
  EXPECT_STREQ("8000098", ce0->station_id()->c_str());
  EXPECT_EQ(2318, ce0->service_num());
  EXPECT_STREQ("", ce0->line_id()->c_str());
  EXPECT_EQ(1444334220, ce0->schedule_time());
  EXPECT_EQ(EventType_ARR, ce0->type());

  auto new_events = inner_msg->new_events();
  ASSERT_EQ(2, new_events->size());

  auto ne0 = new_events->Get(0);
  EXPECT_STREQ("8000118", ne0->base()->base()->station_id()->c_str());
  EXPECT_EQ(2318, ne0->base()->base()->service_num());
  EXPECT_STREQ("", ne0->base()->base()->line_id()->c_str());
  EXPECT_EQ(1444334340, ne0->base()->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, ne0->base()->base()->type());

  EXPECT_STREQ("IC", ne0->base()->category()->c_str());
  EXPECT_STREQ("6", ne0->base()->track()->c_str());

  EXPECT_EQ(RerouteStatus_UmlNeu, ne0->status());

  auto ne1 = new_events->Get(1);
  EXPECT_STREQ("8000118", ne1->base()->base()->station_id()->c_str());
  EXPECT_EQ(2318, ne1->base()->base()->service_num());
  EXPECT_STREQ("", ne1->base()->base()->line_id()->c_str());
  EXPECT_EQ(1444334460, ne1->base()->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, ne1->base()->base()->type());

  EXPECT_STREQ("IC", ne1->base()->category()->c_str());
  EXPECT_STREQ("6", ne1->base()->track()->c_str());

  EXPECT_EQ(RerouteStatus_UmlNeu, ne1->status());
}

constexpr auto const reroute_fixture_only_new = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007000240802" KNr="123865299">
 <ListNachricht>
  <Nachricht>
   <Umleitung>
    <Service Id="85783595" IdZNr="83236" IdZGattung="Bus" IdBf="CCAXU"
             IdBfEvaNr="0492865" IdZeit="20151007003000" ZielBfCode="CCAXV"
             ZielBfEvaNr="0492866" Zielzeit="20151007005500" IdVerwaltung="rabRAB"
             IdZGattungInt="Bus" IdLinie="2" SourceZNr="EFZ">
     <ListZug/>
     <ListUml>
      <Uml>
       <ListZug>
        <Zug Nr="83236" Gattung="Bus" Linie="2" GattungInt="Bus" Verwaltung="rabRAB">
         <ListZE>
          <ZE Typ="Start" RegSta="Normal">
           <Bf Code="CCAXU" EvaNr="0492865" Name="Ravensburg Bahnhof"/>
           <Zeit Soll="20151007003000"/>
          </ZE>
          <ZE Typ="Ziel" RegSta="Normal">
           <Bf Code="CCAXV" EvaNr="0492866" Name="Marsweiler, Baindt"/>
           <Zeit Soll="20151007005500"/>
          </ZE>
         </ListZE>
        </Zug>
       </ListZug>
      </Uml>
     </ListUml>
    </Service>
   </Umleitung>
   <ListQuelle>
    <Quelle Sender="RSL" Typ="Umleitung" KNr="13655100700023900002"
            TIn="20151007000240209" TOutSnd="20151007000240"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_reroute_message, message_only_new) {
  auto const messages = parse(reroute_fixture_only_new);
  ASSERT_EQ(1, messages.size());

  auto outer_msg = GetMessage(messages[0].data());
  ASSERT_EQ(MessageUnion_RerouteMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<RerouteMessage const*>(outer_msg->content());

  auto cancelled_events = inner_msg->cancelled_events();
  ASSERT_EQ(0, cancelled_events->size());

  auto new_events = inner_msg->new_events();
  ASSERT_EQ(2, new_events->size());

  auto ne0 = new_events->Get(0);
  EXPECT_STREQ("2", ne0->base()->base()->line_id()->c_str());

  auto ne1 = new_events->Get(1);
  EXPECT_STREQ("2", ne1->base()->base()->line_id()->c_str());
}

constexpr auto const reroute_fixture_only_cancel = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007000241148" KNr="123865305">
 <ListNachricht>
  <Nachricht>
   <Umleitung>
    <Service Id="85783595" IdZNr="83236" IdZGattung="Bus" IdBf="CCAXU"
             IdBfEvaNr="0492865" IdZeit="20151006003000" ZielBfCode="CCAXV"
             ZielBfEvaNr="0492866" Zielzeit="20151006005500" IdVerwaltung="rabRAB"
             IdZGattungInt="Bus" IdLinie="2" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="83236" Gattung="Bus" Linie="2" GattungInt="Bus" Verwaltung="rabRAB">
       <ListZE>
        <ZE Typ="An" Status="UmlAusf">
         <Bf Code="CCAYM" EvaNr="0492883" Name="Gartenstraße, Ravensburg"/>
         <Zeit Soll="20151006003300"/>
        </ZE>
        <ZE Typ="Ab" Status="UmlAusf">
         <Bf Code="CCAXW" EvaNr="0492867" Name="Schreinerei Dreher, Baindt"/>
         <Zeit Soll="20151006005400"/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
     <ListUml>
      <Uml>
       <ListZug/>
      </Uml>
     </ListUml>
    </Service>
   </Umleitung>
   <ListQuelle>
    <Quelle Sender="RSL" Typ="Umleitung" KNr="13655100700023900002"
            TIn="20151007000240209" TOutSnd="20151007000240"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_reroute_message, message_only_cancel) {
  auto const messages = parse(reroute_fixture_only_cancel);
  ASSERT_EQ(1, messages.size());

  auto outer_msg = GetMessage(messages[0].data());
  ASSERT_EQ(MessageUnion_RerouteMessage, outer_msg->content_type());
  auto inner_msg =
      reinterpret_cast<RerouteMessage const*>(outer_msg->content());

  auto cancelled_events = inner_msg->cancelled_events();
  ASSERT_EQ(2, cancelled_events->size());

  auto ce0 = cancelled_events->Get(0);
  EXPECT_STREQ("2", ce0->line_id()->c_str());

  auto ce1 = cancelled_events->Get(1);
  EXPECT_STREQ("2", ce1->line_id()->c_str());

  auto new_events = inner_msg->new_events();
  ASSERT_EQ(0, new_events->size());
}

}  // namespace motis::ris::risml
