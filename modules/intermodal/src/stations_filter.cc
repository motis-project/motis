#include "motis/intermodal/stations_filter.h"

#include "utl/erase_if.h"

using namespace geo;
using namespace flatbuffers;


namespace motis::intermodal {

struct stations_distances {
  double distance;
  std::string station_id;
  bool dismiss;
};

// links muss das kleinste stehen, die Funktion richtig?
bool distance_sort(stations_distances const& lhs, stations_distances const& rhs) {
    return lhs.distance < rhs.distance;
}

//
//bool when_to_remove(const minimalistic_station& ms, const stations_distances& sd) {
//  return ms.id == sd.station_id && sd.dismiss;
//}

std::vector<minimalistic_station> first_filter(latlng pos, int max_dur, int max_dist, const Vector<Offset<Station>> *stations) {
  int anzahl = stations->size();

  std::vector<minimalistic_station> v_ms;
  std::vector<minimalistic_station> v_filtered_ms;
  std::vector<stations_distances> v_sd;

  latlng station_position;
  for(auto const s : *stations)
  {
    station_position.lat_ = s->pos()->lat();
    station_position.lng_ = s->pos()->lng();
    double dist = distance(pos, station_position);
    stations_distances sd = {dist, s->id()->c_str()};
    v_sd.emplace_back(sd);
    Position motispos_spos = Position(station_position.lat_, station_position.lng_);
    minimalistic_station ms = {motispos_spos, station_position, s->name()->c_str(), s->id()->c_str()};
    v_ms.emplace_back(ms);
  }

  // 15 * 0,8 = 12
  // 15 - 12 - 1 = 2
  // [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
  std::sort(v_sd.begin(), v_sd.end(), &distance_sort);
  int persent_to_dismiss = static_cast<int>(v_sd.size() * 0.8);
  int index = v_sd.size() - persent_to_dismiss - 1;
  for(int t = 0; t < v_sd.size(); t++)
  {
    if(t <= index)
    {
      v_sd.at(t).dismiss = false;
    }
    else
    {
      v_sd.at(t).dismiss = true;
    }
  }

  for(auto & ms : v_ms)
  {
    for(auto & sd : v_sd)
    {
      if(ms.id == sd.station_id && !sd.dismiss)
      {
        v_filtered_ms.emplace_back(ms);
      }
    }
  }

  int deleted = anzahl - v_filtered_ms.size();
  printf("Filter fertig! Gelöscht: %d\n", deleted);
  return v_filtered_ms;
}


} //namespace motis::intermodal