#pragma once

#include "motis/hash_map.h"
#include "motis/string.h"

#include "motis/core/schedule/connection.h"

namespace motis::loader {

inline mcd::hash_map<mcd::string, service_class> class_mapping() {
  return {
      // planes
      {"Flug", service_class::AIR},
      {"Air", service_class::AIR},
      {"International Air", service_class::AIR},
      {"Domestic Air", service_class::AIR},
      {"Intercontinental Air", service_class::AIR},
      {"Domestic Scheduled Air", service_class::AIR},
      {"Shuttle Air", service_class::AIR},
      {"Intercontinental Charter Air", service_class::AIR},
      {"International Charter Air", service_class::AIR},
      {"Round-Trip Charter Air", service_class::AIR},
      {"Sightseeing Air", service_class::AIR},
      {"Helicopter Air", service_class::AIR},
      {"Domestic Charter Air", service_class::AIR},
      {"Schengen-Area Air", service_class::AIR},
      {"Airship", service_class::AIR},
      {"All Airs", service_class::AIR},

      // high speed
      {"High Speed Rail", service_class::ICE},
      {"ICE", service_class::ICE},
      {"THA", service_class::ICE},
      {"TGV", service_class::ICE},
      {"RJ", service_class::ICE},
      {"RJX", service_class::ICE},

      // long range rail
      {"Long Distance Trains", service_class::IC},
      {"Inter Regional Rail", service_class::IC},
      {"Eurocity", service_class::IC},
      {"EC", service_class::IC},
      {"IC", service_class::IC},
      {"EX", service_class::IC},
      {"EXT", service_class::IC},
      {"D", service_class::IC},
      {"InterRegio", service_class::IC},
      {"Intercity", service_class::IC},

      // long range bus
      {"Coach", service_class::COACH},
      {"International Coach", service_class::COACH},
      {"National Coach", service_class::COACH},
      {"Shuttle Coach", service_class::COACH},
      {"Regional Coach", service_class::COACH},
      {"Special Coach", service_class::COACH},
      {"Sightseeing Coach", service_class::COACH},
      {"Tourist Coach", service_class::COACH},
      {"Commuter Coach", service_class::COACH},
      {"All Coachs", service_class::COACH},
      {"EXB", service_class::COACH},  // long-distance bus

      // night trains
      {"Sleeper Rail", service_class::N},
      {"CNL", service_class::N},
      {"EN", service_class::N},
      {"Car Transport Rail", service_class::N},
      {"Lorry Transport Rail", service_class::N},
      {"Vehicle Transport Rail", service_class::N},
      {"AZ", service_class::N},
      {"NJ", service_class::N},  // Night Jet

      // fast local trains
      {"RE", service_class::RE},
      {"REX", service_class::RE},
      {"IR", service_class::RE},
      {"IRE", service_class::RE},
      {"X", service_class::RE},
      {"DPX", service_class::RE},
      {"E", service_class::RE},
      {"Sp", service_class::RE},
      {"RegioExpress", service_class::RE},
      {"TER", service_class::RE},  // Transport express regional
      {"TE2", service_class::RE},  // Transport express regional
      {"Cross-Country Rail", service_class::RE},

      // local trains
      {"Railway Service", service_class::RB},
      {"Regional Rail", service_class::RB},
      {"Tourist Railway", service_class::RB},
      {"Rail Shuttle (Within Complex)", service_class::RB},
      {"Replacement Rail", service_class::RB},
      {"Special Rail", service_class::RB},
      {"Rack and Pinion Railway", service_class::RB},
      {"Additional Rail", service_class::RB},
      {"All Rails", service_class::RB},
      {"DPN", service_class::RB},
      {"R", service_class::RB},
      {"DPF", service_class::RB},
      {"RB", service_class::RB},
      {"Os", service_class::RB},
      {"Regionalzug", service_class::RB},
      {"RZ", service_class::RB},
      {"CC", service_class::RB},  // narrow-gauge mountain train
      {"PE", service_class::RB},  // Panorama Express

      // metro
      {"S", service_class::S},
      {"S-Bahn", service_class::S},
      {"SB", service_class::S},
      {"Metro", service_class::S},
      {"Schnelles Nachtnetz", service_class::S},
      {"SN", service_class::S},  // S-Bahn Nachtlinie

      // subway
      {"U", service_class::U},
      {"STB", service_class::U},
      {"M", service_class::U},  // subway in Lausanne

      // street-car
      {"Tram", service_class::STR},
      {"STR", service_class::STR},
      {"Str", service_class::STR},
      {"T", service_class::STR},

      // bus
      {"Bus", service_class::BUS},
      {"B", service_class::BUS},
      {"BN", service_class::BUS},
      {"BP", service_class::BUS},
      {"CAR", service_class::BUS},
      {"KB", service_class::BUS},

      // ship
      {"Schiff", service_class::SHIP},
      {"Fähre", service_class::SHIP},
      {"BAT", service_class::SHIP},  // "bateau"
      {"KAT", service_class::SHIP},
      {"Ferry", service_class::SHIP},
      {"Water Transport", service_class::SHIP},
      {"International Car Ferry", service_class::SHIP},
      {"National Car Ferry", service_class::SHIP},
      {"Regional Car Ferry", service_class::SHIP},
      {"Local Car Ferry", service_class::SHIP},
      {"International Passenger Ferry", service_class::SHIP},
      {"National Passenger Ferry", service_class::SHIP},
      {"Regional Passenger Ferry", service_class::SHIP},
      {"Local Passenger Ferry", service_class::SHIP},
      {"Post Boat", service_class::SHIP},
      {"Train Ferry", service_class::SHIP},
      {"Road-Link Ferry", service_class::SHIP},
      {"Airport-Link Ferry", service_class::SHIP},
      {"Car High-Speed Ferry", service_class::SHIP},
      {"Passenger High-Speed Ferry", service_class::SHIP},
      {"Sightseeing Boat", service_class::SHIP},
      {"School Boat", service_class::SHIP},
      {"Cable-Drawn Boat", service_class::SHIP},
      {"River Bus", service_class::SHIP},
      {"Scheduled Ferry", service_class::SHIP},
      {"Shuttle Ferry", service_class::SHIP},
      {"All Water Transports", service_class::SHIP},

      // other
      {"ZahnR", service_class::OTHER},
      {"Schw-B", service_class::OTHER},
      {"EZ", service_class::OTHER},
      {"Taxi", service_class::OTHER},
      {"ALT", service_class::OTHER},  // "Anruflinientaxi"
      {"AST", service_class::OTHER},  // "Anrufsammeltaxi"
      {"RFB", service_class::OTHER},
      {"RT", service_class::OTHER},
      {"Taxi", service_class::OTHER},
      {"Communal Taxi", service_class::OTHER},
      {"Water Taxi", service_class::OTHER},
      {"Rail Taxi", service_class::OTHER},
      {"Bike Taxi", service_class::OTHER},
      {"Licensed Taxi", service_class::OTHER},
      {"Private Hire Vehicle", service_class::OTHER},
      {"All Taxis", service_class::OTHER},
      {"Self Drive", service_class::OTHER},
      {"Hire Car", service_class::OTHER},
      {"Hire Van", service_class::OTHER},
      {"Hire Motorbike", service_class::OTHER},
      {"Hire Cycle", service_class::OTHER},
      {"All Self-Drive Vehicles", service_class::OTHER},
      {"Car train", service_class::OTHER},
      {"GB", service_class::OTHER},  // ski lift / "funicular"?
      {"PB", service_class::OTHER},  // also a ski lift(?)
      {"FUN", service_class::OTHER},  // "funicular"
      {"Funicular", service_class::OTHER},
      {"Telecabin", service_class::OTHER},
      {"Cable Car", service_class::OTHER},
      {"Chair Lift", service_class::OTHER},
      {"Drag Lift", service_class::OTHER},
      {"Small Telecabin", service_class::OTHER},
      {"All Telecabins", service_class::OTHER},
      {"All Funicular", service_class::OTHER},
      {"Drahtseilbahn", service_class::OTHER},
      {"Standseilbahn", service_class::OTHER},
      {"Sesselbahn", service_class::OTHER},
      {"Gondola, Suspended cable car", service_class::OTHER},
      {"Aufzug", service_class::OTHER},
      {"Elevator", service_class::OTHER},
      {"ASC", service_class::OTHER}  // some elevator in Bern
  };
}

}  // namespace motis::loader
