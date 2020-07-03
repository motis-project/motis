#pragma once

#include "motis/hash_map.h"
#include "motis/string.h"

#include "motis/core/schedule/connection.h"

namespace motis::loader {

inline mcd::hash_map<mcd::string, service_class> class_mapping() {
  return {
      // planes
      {"Flug", service_class::AIR},

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
      {"EXB", service_class::COACH},  // long-distance bus

      // night trains
      {"Sleeper Rail", service_class::N},
      {"CNL", service_class::N},
      {"EN", service_class::N},
      {"Car Transport Rail", service_class::N},
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

      // local trains
      {"Railway Service", service_class::RB},
      {"Regional Rail", service_class::RB},
      {"Tourist Railway", service_class::RB},
      {"Rail Shuttle (Within Complex)", service_class::RB},
      {"DPN", service_class::RB},
      {"R", service_class::RB},
      {"DPF", service_class::RB},
      {"RB", service_class::RB},
      {"Os", service_class::RB},
      {"Regionalzug", service_class::RB},
      {"RZ", service_class::RB},
      {"CC", service_class::RB},  // narrow-gauge mountain train
      {"PE", service_class::RB},  // Panorama Express
      {"T", service_class::RB},

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

      // other
      {"ZahnR", service_class::OTHER},
      {"Schw-B", service_class::OTHER},
      {"EZ", service_class::OTHER},
      {"Taxi", service_class::OTHER},
      {"ALT", service_class::OTHER},  // "Anruflinientaxi"
      {"AST", service_class::OTHER},  // "Anrufsammeltaxi"
      {"RFB", service_class::OTHER},
      {"RT", service_class::OTHER},
      {"GB", service_class::OTHER},  // ski lift / "funicular"?
      {"PB", service_class::OTHER},  // also a ski lift(?)
      {"FUN", service_class::OTHER},  // "funicular"
      {"Drahtseilbahn", service_class::OTHER},
      {"Standseilbahn", service_class::OTHER},
      {"Sesselbahn", service_class::OTHER},
      {"Aufzug", service_class::OTHER},
      {"ASC", service_class::OTHER}  // some elevator in Bern
  };
}

}  // namespace motis::loader
