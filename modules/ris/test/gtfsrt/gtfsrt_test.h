#pragma once

#include "gtest/gtest.h"

#include <string>
#include "google/protobuf/util/json_util.h"

#include "motis/core/schedule/schedule.h"
#include "motis/loader/loader_options.h"

#include "motis/protocol/RISMessage_generated.h"
#include "gtfsrt.pb.h"

namespace motis::ris {
struct ris_message;
namespace gtfsrt {

inline std::string json_to_protobuf(std::string msg_json) {
  transit_realtime::FeedMessage msg;
  google::protobuf::util::JsonStringToMessage(msg_json, &msg);
  auto binary_msg = msg.SerializeAsString();
  return binary_msg;
};

class gtfsrt_test : public ::testing::Test {
protected:
  gtfsrt_test(loader::loader_options const&);

  void SetUp() override;

  std::vector<ris_message> parse_json(std::string const&);

  schedule_ptr sched_;
  loader::loader_options opts_;
};

}  // namespace gtfsrt
}  // namespace motis::ris