#include "gtest/gtest.h"

#include "motis/protocol/RISMessage_generated.h"
#include "motis/ris/risml/risml_parser.h"

using namespace motis::ris::risml;
using namespace motis::ris;
using namespace motis;

constexpr auto ist_fixture_1 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151006235934781" KNr="123863714">
 <ListNachricht>
  <Nachricht>
   <Ist>
    <Service Id="85713913" IdZNr="8329" IdZGattung="S" IdBf="MMAM"
             IdBfEvaNr="8004204" IdZeit="20151006234400"  ZielBfCode="MHO"
             ZielBfEvaNr="8002980" Zielzeit="20151007010600" IdVerwaltung="07"
             IdZGattungInt="s" IdLinie="3" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="8329" Gattung="S" Linie="3"  GattungInt="s" Verwaltung="07" >
       <ListZE>
        <ZE Typ="Ab">
         <Bf Code="MOL" EvaNr="8004667" Name="Olching" />
         <Zeit Soll="20151006235900" Ist="20151006235900" />
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ist>
   <ListQuelle>
    <Quelle Sender="ZENTRAL"  Typ="IstProg" KNr="18762" TIn="20151006235920176"
            TOutSnd="20151006235934696"/>
    <Quelle Sender="10.35.205.140:7213/13" Typ="UIC 102" TIn="20151006235933"
            Esc="mue810jyct" />
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_delay_message, ist_message_1) {
  auto const messages = parse(ist_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168774, message.timestamp_);
  EXPECT_EQ(1444172760, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  auto id = inner_msg->trip_id();
  EXPECT_STREQ("8004204", id->station_id()->c_str());
  EXPECT_EQ(8329, id->service_num());
  EXPECT_EQ(1444167840, id->schedule_time());
  EXPECT_EQ(IdEventType_Schedule, id->trip_type());

  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(1, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8004667", e0->base()->station_id()->c_str());

  EXPECT_EQ(8329, e0->base()->service_num());
  EXPECT_STREQ("3", e0->base()->line_id()->c_str());
  EXPECT_EQ(1444168740, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());

  EXPECT_EQ(1444168740, e0->updated_time());
}

constexpr auto ist_fixture_2 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007000002316" KNr="123863963">
 <ListNachricht>
  <Nachricht>
   <Ist>
    <Service Id="86009330" IdZNr="60418" IdZGattung="IC" IdBf="MH"
             IdBfEvaNr="8000261" IdZeit="20151006225000"  ZielBfCode="TS"
             ZielBfEvaNr="8000096" Zielzeit="20151007011600" IdVerwaltung="80"
             IdZGattungInt="IC" IdLinie="" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="60418" Gattung="IC"  GattungInt="IC" Verwaltung="80" >
       <ListZE>
        <ZE Typ="Ab" >
         <Bf Code="MOL" EvaNr="8004667" />
         <Zeit Soll="20151006234800" Ist="20151006235900" />
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ist>
   <ListQuelle>
    <Quelle Sender="ZENTRAL"  Typ="IstProg" KNr="18777" TIn="20151007000000336"
            TOutSnd="20151007000002053"/>
    <Quelle Sender="10.35.205.140:7213/13"
            Typ="UIC 102" TIn="20151007000001" Esc="mue810jyhi" />
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_delay_message, ist_message_2) {
  auto const messages = parse(ist_fixture_2);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168802, message.timestamp_);
  EXPECT_EQ(1444173360, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(1, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8004667", e0->base()->station_id()->c_str());

  EXPECT_EQ(60418, e0->base()->service_num());
  EXPECT_STREQ("", e0->base()->line_id()->c_str());
  EXPECT_EQ(1444168080, e0->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e0->base()->type());

  EXPECT_EQ(1444168740, e0->updated_time());
}

constexpr auto ist_fixture_3 = R"(
<?xml version="1.0"?>
<Paket Version="1.2" SpezVer="1" TOut="20151116171000721" KNr="187859610">
 <ListNachricht>
  <Nachricht>
   <Ist>
    <Service Id="249933654442" IdZGattung="RB" IdZGattungInt="RB"
             IdBf="MKCH" IdBfEvaNr="8003355" IdZeit="20151116164500" IdZNr="59622"
             ZielBfCode="MH  N" ZielBfEvaNr="8098261" Zielzeit="20151116180000"
             IdVerwaltung="07" SourceZNr="EFZ" RegSta="Sonderzug">
     <ListZug>
      <Zug Nr="59622" Gattung="RB" GattungInt="RB" Name="RB 59622" Verwaltung="07">
       <ListZE>
        <ZE Typ="An">
         <Bf Code="MSTA" EvaNr="8005672" Name="Iffeldorf"/>
         <Zeit Soll="20151116170800" Ist="20151116170900"/>
        </ZE>
        <ZE Typ="Ab">
         <Bf Code="MSTA" EvaNr="8005672" Name="Iffeldorf"/>
         <Zeit Soll="20151116170900" Ist="20151116170900"/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ist>
   <ListQuelle>
    <Quelle Sender="ZENTRAL" Typ="IstProg" KNr="3325"
            TIn="20151116170958659" TOutSnd="20151116170958601"/>
    <Quelle Sender="GPS" TIn="20151116170954"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_delay_message, ist_message_3) {
  auto const messages = parse(ist_fixture_3);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1447690200, message.timestamp_);
  EXPECT_EQ(1447688700, message.earliest_);
  EXPECT_EQ(1447693200, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  EXPECT_EQ(DelayType_Is, inner_msg->type());

  auto id = inner_msg->trip_id();
  EXPECT_EQ(IdEventType_Additional, id->trip_type());

  auto events = inner_msg->events();
  ASSERT_EQ(2, events->size());

  auto const& e0 = events->Get(0);
  EXPECT_STREQ("8005672", e0->base()->station_id()->c_str());

  EXPECT_EQ(59622, e0->base()->service_num());
  EXPECT_STREQ("", e0->base()->line_id()->c_str());
  EXPECT_EQ(1447690080, e0->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->base()->type());
  EXPECT_EQ(1447690140, e0->updated_time());

  auto const& e1 = events->Get(1);
  EXPECT_STREQ("8005672", e1->base()->station_id()->c_str());

  EXPECT_EQ(59622, e1->base()->service_num());
  EXPECT_STREQ("", e1->base()->line_id()->c_str());
  EXPECT_EQ(1447690140, e1->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->base()->type());
  EXPECT_EQ(1447690140, e1->updated_time());
}

std::string type_fixture(std::string const& type_string) {
  return std::string(R"(<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket TOut="20151116180000">
 <ListNachricht>
  <Nachricht>
   <Ist>
    <Service Id="249933654442" IdZGattung="RB" IdZGattungInt="RB"
             IdBf="MKCH" IdBfEvaNr="8003355" IdZeit="20151116164500" IdZNr="59622"
             ZielBfCode="MH  N" ZielBfEvaNr="8098261" Zielzeit="20151116180000"
             IdVerwaltung="07" SourceZNr="EFZ">
     <ListZug>
      <Zug>
       <ListZE>
        <ZE Typ=")") +
         type_string + R"(">
         <Bf/>
         <Zeit Soll="20151116180000" Ist="20151116180000"/>
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </Ist>
   <ListQuelle/>
  </Nachricht>
 </ListNachricht>
</Paket>
)";
}

EventType get_type(std::vector<ris_message> const& messages) {
  if (messages.empty()) {
    throw std::runtime_error("messages empty");
  }
  auto content = motis::ris::GetMessage(messages[0].data())->content();
  auto delay_message = reinterpret_cast<DelayMessage const*>(content);
  return delay_message->events()->Get(0)->base()->type();
}

TEST(ris_delay_message, train_event_type) {
  auto start_msg = type_fixture("Start");
  auto start = parse(start_msg.c_str());
  ASSERT_EQ(EventType_DEP, get_type(start));

  auto ab_msg = type_fixture("Ab");
  auto ab = parse(ab_msg.c_str());
  ASSERT_EQ(EventType_DEP, get_type(ab));

  auto an_msg = type_fixture("An");
  auto an = parse(an_msg.c_str());
  ASSERT_EQ(EventType_ARR, get_type(an));

  auto ziel_msg = type_fixture("Ziel");
  auto ziel = parse(ziel_msg.c_str());
  ASSERT_EQ(EventType_ARR, get_type(ziel));

  // "Durch" events are ignored
  auto pass_msg = type_fixture("Durch");
  auto pass = parse(pass_msg.c_str());
  auto content = motis::ris::GetMessage(pass[0].data())->content();
  auto delay_message = reinterpret_cast<DelayMessage const*>(content);
  EXPECT_EQ(0, delay_message->events()->size());
}

constexpr auto ist_prog_fixture_1 = R"(
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151006235943156" KNr="123863771">
 <ListNachricht>
  <Nachricht>
   <IstProg >
    <Service Id="85825746" IdZNr="21839" IdZGattung="RE" IdBf="AL"
             IdBfEvaNr="8000237" IdZeit="20151006232300"  ZielBfCode="ABCH"
             ZielBfEvaNr="8000058" Zielzeit="20151007000500" IdVerwaltung="02"
             IdZGattungInt="RE" IdLinie="" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="21839" Gattung="RE"  GattungInt="RE" Verwaltung="02" >
       <ListZE>
        <ZE Typ="Ziel" >
         <Bf Code="ABCH" EvaNr="8000058" Name="B�chen"/>
         <Zeit Soll="20151007000500" Prog="20151007000600" Dispo="NEIN" />
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </IstProg>
   <ListQuelle>
    <Quelle Sender="ZENTRAL"  Typ="IstProg" KNr="18767"
            TIn="20151006235939097" TOutSnd="20151006235943107"/>
    <Quelle Sender="10.35.204.12:7213/13" Typ="UIC 102" TIn="20151006235942"
            Esc="bln810jye7" />
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
)";

TEST(ris_delay_message, ist_prog_message_1) {
  auto const messages = parse(ist_prog_fixture_1);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168783, message.timestamp_);
  EXPECT_EQ(1444166580, message.earliest_);
  EXPECT_EQ(1444169100, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  EXPECT_EQ(DelayType_Forecast, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(1, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8000058", e0->base()->station_id()->c_str());

  EXPECT_EQ(21839, e0->base()->service_num());
  EXPECT_EQ(1444169100, e0->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->base()->type());

  EXPECT_EQ(1444169160, e0->updated_time());
}

constexpr auto ist_prog_fixture_2 = R"--((
<?xml version="1.0" encoding="iso-8859-1" ?>
<Paket Version="1.2" SpezVer="1" TOut="20151007000000657" KNr="123863949">
 <ListNachricht>
  <Nachricht>
   <IstProg >
    <Service Id="85712577" IdZNr="37616" IdZGattung="S" IdBf="APB"
             IdBfEvaNr="8004862" IdZeit="20151006231000"  ZielBfCode="AWL"
             ZielBfEvaNr="8006236" Zielzeit="20151007001900" IdVerwaltung="0S"
             IdZGattungInt="s" IdLinie="1" SourceZNr="EFZ">
     <ListZug>
      <Zug Nr="37616" Gattung="S" Linie="1"  GattungInt="s" Verwaltung="0S" >
       <ListZE>
        <ZE Typ="An" >
         <Bf Code="ARI" EvaNr="8005106" Name="Hamburg-Rissen"/>
         <Zeit Soll="20151007001400" Prog="20151007001800" Dispo="NEIN" />
        </ZE>
        <ZE Typ="Ab" >
         <Bf Code="ARI" EvaNr="8005106" Name="Hamburg-Rissen"/>
         <Zeit Soll="20151007001400" Prog="20151007001800" Dispo="NEIN" />
        </ZE>
        <ZE Typ="Ziel" >
         <Bf Code="AWL" EvaNr="8006236" Name="Wedel(Holst)"/>
         <Zeit Soll="20151007001900" Prog="20151007002200" Dispo="NEIN" />
        </ZE>
       </ListZE>
      </Zug>
     </ListZug>
    </Service>
   </IstProg>
   <ListQuelle>
    <Quelle Sender="SBahnHamburg"  Typ="IstProg" KNr="8909333"
            TIn="20151006235950236" TOutSnd="20151007000000064"/>
   </ListQuelle>
  </Nachricht>
 </ListNachricht>
</Paket>
))--";

TEST(ris_delay_message, ist_prog_message_2) {
  auto const messages = parse(ist_prog_fixture_2);
  ASSERT_EQ(1, messages.size());

  auto const& message = messages[0];
  EXPECT_EQ(1444168800, message.timestamp_);
  EXPECT_EQ(1444165800, message.earliest_);
  EXPECT_EQ(1444169940, message.latest_);

  auto outer_msg = motis::ris::GetMessage(message.data());
  ASSERT_EQ(MessageUnion_DelayMessage, outer_msg->content_type());
  auto inner_msg = reinterpret_cast<DelayMessage const*>(outer_msg->content());

  EXPECT_EQ(DelayType_Forecast, inner_msg->type());

  auto events = inner_msg->events();
  ASSERT_EQ(3, events->size());

  auto e0 = events->Get(0);
  EXPECT_STREQ("8005106", e0->base()->station_id()->c_str());

  EXPECT_EQ(37616, e0->base()->service_num());
  EXPECT_STREQ("1", e0->base()->line_id()->c_str());
  EXPECT_EQ(1444169640, e0->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e0->base()->type());
  EXPECT_EQ(1444169880, e0->updated_time());

  auto e1 = events->Get(1);
  EXPECT_STREQ("8005106", e1->base()->station_id()->c_str());

  EXPECT_EQ(37616, e1->base()->service_num());
  EXPECT_STREQ("1", e1->base()->line_id()->c_str());
  EXPECT_EQ(1444169640, e1->base()->schedule_time());
  EXPECT_EQ(EventType_DEP, e1->base()->type());
  EXPECT_EQ(1444169880, e1->updated_time());

  auto e2 = events->Get(2);
  EXPECT_STREQ("8006236", e2->base()->station_id()->c_str());

  EXPECT_EQ(37616, e2->base()->service_num());
  EXPECT_STREQ("1", e2->base()->line_id()->c_str());
  EXPECT_EQ(1444169940, e2->base()->schedule_time());
  EXPECT_EQ(EventType_ARR, e2->base()->type());
  EXPECT_EQ(1444170120, e2->updated_time());
}
