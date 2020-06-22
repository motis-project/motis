#include "gtest/gtest.h"

#include "motis/bootstrap/import_files.h"

namespace mb = motis::bootstrap;

TEST(import_files, make_file_event) {
  using namespace motis;
  using motis::import::FileEvent;

  {
    auto const msg = mb::make_file_event({});
    auto const* fe = motis_content(FileEvent, msg);

    EXPECT_EQ(0, fe->paths()->size());
  }
  {
    auto const msg =
        mb::make_file_event({"schedule-asd:test/schedule/simple_realtime"});
    auto const* fe = motis_content(FileEvent, msg);

    ASSERT_EQ(1, fe->paths()->size());

    auto const* p = fe->paths()->Get(0);
    EXPECT_EQ("schedule", p->tag()->str());
    EXPECT_EQ("asd", p->options()->str());
    EXPECT_EQ("test/schedule/simple_realtime", p->path()->str());
  }
  {
    auto const msg = mb::make_file_event({"cmake:CMakeLists.txt"});
    auto const* fe = motis_content(FileEvent, msg);

    ASSERT_EQ(1, fe->paths()->size());

    auto const* p = fe->paths()->Get(0);
    EXPECT_EQ("cmake", p->tag()->str());
    EXPECT_EQ("", p->options()->str());
    EXPECT_EQ("CMakeLists.txt", p->path()->str());
  }
}
