#pragma once

#include "motis/fwd.h"

namespace nigiri {
struct station_paths;
}  // namespace nigiri

namespace motis {

// Blocks the station graph footpaths and entrance paths (GTFS pathways, ISA)
// whose paths all need an elevator that is out of service now: pathway
// elevators match the FaSta / SIRI-FM elevators within a radius. Routed
// footpaths: time-dependent footpaths (update_rtt_td_footpaths).
void block_station_elevators(nigiri::timetable const&,
                             nigiri::station_paths const*,
                             elevators const&,
                             nigiri::rt_timetable&);

}  // namespace motis
