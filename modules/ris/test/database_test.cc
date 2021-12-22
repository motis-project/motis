#include "gtest/gtest.h"

#include <string>

#include "motis/module/message.h"
#include "motis/test/motis_instance_test.h"
#include "motis/test/schedule/simple_realtime.h"

using namespace flatbuffers;
using namespace motis::test;
using namespace motis::test::schedule;
using namespace motis::module;
using motis::test::schedule::simple_realtime::dataset_opt_long;

namespace motis::ris {

struct ris_db_order : public motis_instance_test {
  ris_db_order()
      : motis::test::motis_instance_test(
            dataset_opt_long, {"ris"},
            {"--ris.input="
             "modules/ris/test_resources/database_test/order_test"}) {}

  static msg_ptr forward(time_t time) {
    message_creator fbb;
    fbb.create_and_finish(MsgContent_RISForwardTimeRequest,
                          CreateRISForwardTimeRequest(fbb, time).Union(),
                          "/ris/forward");
    return make_msg(fbb);
  }
};

TEST_F(ris_db_order, order_all_at_once) {
  std::vector<msg_ptr> msgs;
  subscribe("/ris/messages", msg_sink(&msgs));
  subscribe("/ris/system_time_changed", msg_sink(&msgs));
  call(forward(unix_time(1206)));

  ASSERT_EQ(2U, msgs.size());
  ASSERT_EQ(2U, motis_content(RISBatch, msgs[0])->messages()->size());
  EXPECT_EQ(unix_time(1205), motis_content(RISBatch, msgs[0])
                                 ->messages()
                                 ->Get(0)
                                 ->message_nested_root()
                                 ->timestamp());
  EXPECT_EQ(unix_time(1206), motis_content(RISBatch, msgs[0])
                                 ->messages()
                                 ->Get(1)
                                 ->message_nested_root()
                                 ->timestamp());
  EXPECT_STREQ("/ris/system_time_changed",
               msgs[1]->get()->destination()->target()->c_str());

  msgs.clear();

  call(forward(unix_time(1207)));
  ASSERT_EQ(2U, msgs.size());
  ASSERT_EQ(1U, motis_content(RISBatch, msgs[0])->messages()->size());
  EXPECT_EQ(unix_time(1207), motis_content(RISBatch, msgs[0])
                                 ->messages()
                                 ->Get(0)
                                 ->message_nested_root()
                                 ->timestamp());
  EXPECT_STREQ("/ris/system_time_changed",
               msgs[1]->get()->destination()->target()->c_str());
}

}  // namespace motis::ris
