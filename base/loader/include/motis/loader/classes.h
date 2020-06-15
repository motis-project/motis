#pragma once

#include "motis/hash_map.h"
#include "motis/string.h"

namespace motis::loader {

inline mcd::hash_map<mcd::string, uint8_t> class_mapping() {
  // clang-format off
  return {
    // high speed
    { "High Speed Rail", 0},
    { "ICE", 0},
    { "THA", 0},
    { "TGV", 0},
    { "RJ", 0},

    // long range
    {"Long Distance Trains", 1},
    {"Inter Regional Rail", 1},
    {"Eurocity", 1},
    { "EC", 1 },
    { "IC", 1 },
    { "EX", 1 },
    { "EXT", 1 },
    { "D", 1 },
    { "IR", 1},
    { "InterRegio", 1},
    { "Intercity", 1},

    // night trains
    { "Sleeper Rail", 2},
    { "CNL", 2 },
    { "EN", 2 },
    { "Car Transport Rail", 2},
    { "AZ", 2 },
    { "NJ", 2 },

    // fast local trains
    { "IRE", 3 },
    { "REX", 3 },
    { "RE", 3 },
    { "IR", 3 },
    { "X", 3 },
    { "DPX", 3 },
    { "E", 3 },
    { "Sp", 3 },
    { "RegioExpress", 3},

    // local trains
    { "Regional Rail", 4 },
    { "Railway Service", 4 },
    { "Tourist Railway", 4 },
    { "Rail Shuttle (Within Complex)", 4 },
    { "DPN", 4 },
    { "R", 4 },
    { "DPF", 4 },
    { "RB", 4 },
    { "Os", 4 },
    { "Regionalzug", 4 },
    { "RZ", 4 },
    { "CC", 4}, // narrow-gauge mountain train

    // metro
    { "S", 5 },
    { "S-Bahn", 5},
    { "SB", 5 },
    { "Metro", 5 },
    { "Schnelles Nachtnetz", 5},

    // subway
    { "U", 6 },
    { "STB", 6 },
    { "M", 6 },  // subway in Lausanne

    // street-car
    { "Tram", 7 },
    { "STR", 7 },
    { "Str", 7 },
    { "T", 7 },

    // bus
    { "Bus", 8 },
    { "B", 8 },
    { "BN", 8 },
    { "BP", 8 },
    { "CAR", 8 },
    { "EXB", 8 }, // long-distance bus
    { "KB", 8 },

    // other
    { "Flug", 9 },
    { "Schiff", 9 },
    { "ZahnR", 9 },
    { "Schw-B", 9 },
    { "Fähre", 9 },
    { "BAT", 9 }, // "bateau"
    { "KAT", 9 },
    { "EZ", 9 },
    { "ALT", 9 },
    { "AST", 9 },
    { "RFB", 9 },
    { "RT", 9 },
    {"Drahtseilbahn", 9},
    { "GB", 9 }, // ski lift?
    {"Standseilbahn", 9},
    {"FUN", 9}, // "funicular"
    {"GB", 9}, // "funicular"
    {"Sesselbahn", 9},
    {"Taxi", 9},
    {"Aufzug", 9},
    {"ASC", 9}, // some elevator in Bern
    {"Schiff", 9}
  };
  // clang-format on
}

}  // namespace motis::loader
