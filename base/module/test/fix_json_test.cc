#include "gtest/gtest.h"

#include "motis/module/message.h"

using namespace motis::module;

auto const req = std::string{R"({
  "destination": { "target": "/intermodal" },
  "content": {
    "search_type": "Default",
    "start": {
      "position": { "lat": 49.87743560612768, "lng": 8.654404878616335 },
      "interval": { "begin": 1534747500, "end": 1534754700 }
    },
    "start_modes": [
      {
        "mode_type": "FootPPR",
        "mode": {
          "search_options": { "profile": "default", "duration_limit": 900 }
        }
      }
    ],
    "start_type": "IntermodalPretripStart",
    "destination": { "lat": 49.87397851823742, "lng": 8.641862869262697 },
    "destination_modes": [
      {
        "mode_type": "FootPPR",
        "mode": {
          "search_options": { "profile": "default", "duration_limit": 900 }
        }
      }
    ],
    "search_dir": "Forward",
    "destination_type": "InputPosition",
    "router": ""
  },
  "content_type": "IntermodalRoutingRequest"
})"};

using namespace motis;
using namespace motis::intermodal;

TEST(fix_json, fix_json) {
  EXPECT_THROW(make_msg(req, false), std::system_error);  // NO_LINT
  EXPECT_NO_THROW(make_msg(req, true));  // NO_LINT

  auto const msg = make_msg(req, true);
  auto const r = motis_content(IntermodalRoutingRequest, msg);
  EXPECT_EQ(1, msg->id());
  EXPECT_EQ("/intermodal", msg->get()->destination()->target()->str());

  ASSERT_EQ(IntermodalStart_IntermodalPretripStart, r->start_type());
  auto const start =
      reinterpret_cast<IntermodalPretripStart const*>(r->start());
  EXPECT_EQ(1534747500, start->interval()->begin());
  EXPECT_EQ(1534754700, start->interval()->end());
  EXPECT_EQ(49.87743560612768, start->position()->lat());
  EXPECT_EQ(8.654404878616335, start->position()->lng());
  ASSERT_EQ(1, r->start_modes()->size());
  ASSERT_EQ(Mode_FootPPR, r->start_modes()->Get(0)->mode_type());
  EXPECT_EQ(900,
            reinterpret_cast<FootPPR const*>(r->start_modes()->Get(0)->mode())
                ->search_options()
                ->duration_limit());
  ASSERT_EQ(IntermodalDestination_InputPosition, r->destination_type());
  EXPECT_EQ(49.87397851823742,
            reinterpret_cast<InputPosition const*>(r->destination())->lat());
  EXPECT_EQ(8.641862869262697,
            reinterpret_cast<InputPosition const*>(r->destination())->lng());
}
