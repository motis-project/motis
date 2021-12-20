#include "motis/ris/gtfs-rt/util.h"

#include "utl/verify.h"

#include "google/protobuf/util/json_util.h"
#include "gtfsrt.pb.h"

namespace motis::ris::gtfsrt {

std::string json_to_protobuf(std::string const& json) {
  transit_realtime::FeedMessage msg;
  google::protobuf::util::JsonStringToMessage(json, &msg);
  return msg.SerializeAsString();
}

std::string protobuf_to_json(std::string const& protobuf) {
  transit_realtime::FeedMessage feed_message;
  auto const success = feed_message.ParseFromArray(
      reinterpret_cast<void const*>(protobuf.data()), protobuf.size());
  utl::verify(success, "unable to parse GTFS-RT protobuf message");
  std::string output;
  google::protobuf::util::JsonPrintOptions options;
  options.add_whitespace = true;
  google::protobuf::util::MessageToJsonString(feed_message, &output, options);
  return output;
}

}  // namespace motis::ris::gtfsrt