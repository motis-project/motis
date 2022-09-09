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
  unixtime pickupTime = -1;
  unixtime dropoffTime = -1;
  double price = -1.0;
  std::vector<int> walkDur;  // start, ziel; in sekunden
};

struct availability_request {
  bool start = false;
  std::string productID = "string";
  int64_t duration = -1; // in sekunden
  geo::latlng startpoint = {-1.0, -1.0};
  geo::latlng endpoint = {-1.0, -1.0};
  unixtime departureTime = -1;
  unixtime arrivalTime = -1;
  unixtime arrivalTime_onnext = -1;
  int maxWalkDist = 500; // meter
};

availability_response check_od_availability(availability_request);

} //namespace motis::intermodal