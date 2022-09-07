#include "motis/core/common/unixtime.h"

using namespace std;

namespace motis::intermodal {

struct Dot {
  double lat = -1.0;
  double lng = -1.0;
};

struct availability_response {
  bool available = false;
  string codenumber_id = " ";
  Dot startpoint = {-1.0, -1.0};
  Dot endpoint = {-1.0, -1.0};
  unixtime pickupTime[3] = {-1, -1, -1};
  unixtime dropoffTime[3] = {-1, -1, -1};
  double price = -1.0;
  int walkDur[2] = {-1, -1};  // start, ziel; in sekunden
};

struct availability_request {
  bool start = false;
  string productID = "string";
  int64_t duration = -1; // in sekunden
  Dot startpoint = {-1.0, -1.0};
  Dot endpoint = {-1.0, -1.0};
  unixtime departureTime = -1;
  unixtime arrivalTime = -1;
  unixtime arrivalTime_onnext = -1;
  int maxWalkDist = 500; // meter
};

availability_response check_od_availability(availability_request);

} //namespace motis::intermodal