#pragma once

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include "google/protobuf/util/json_util.h"

#include "motis/core/schedule/schedule.h"
#include "motis/loader/loader_options.h"
#include "motis/ris/gtfs-rt/common.h"

#include "motis/protocol/RISMessage_generated.h"
#include "gtfsrt.pb.h"

namespace motis::ris {

struct ris_message;

namespace gtfsrt {

struct gtfsrt_test : public ::testing::Test {
  gtfsrt_test(loader::loader_options);

  void SetUp() override;

  std::vector<ris_message> parse_json(std::string const&) const;

  schedule_ptr sched_;
  std::unique_ptr<knowledge_context> knowledge_;
  loader::loader_options opts_;
};

}  // namespace gtfsrt
}  // namespace motis::ris
