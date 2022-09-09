#include <string>
#include <vector>

#include "motis/core/common/unixtime.h"
#include "geo/latlng.h"

namespace motis::intermodal {

struct availability_response {
  bool available = false;
  std::string codenumber_id = " ";
  geo::latlng startpoint = {-1.0, -1.0};
  geo::latlng endpoint = {-1.0, -1.0};
  unixtime pickup_time = -1;
  unixtime dropoff_time = -1;
  double price = -1.0;
  std::vector<int> walk_dur;  // start, ziel; in sekunden
};

struct availability_request {
  bool start = false;
  std::string product_id = "string";
  int64_t duration = -1; // in sekunden
  geo::latlng startpoint = {-1.0, -1.0};
  geo::latlng endpoint = {-1.0, -1.0};
  unixtime departure_time = -1;
  unixtime arrival_time = -1;
  unixtime arrival_time_onnext = -1;
  int max_walk_dist = 500; // meter
};

availability_response check_od_availability(availability_request);

} //namespace motis::intermodal