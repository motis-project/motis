#include <filesystem>

#include "google/protobuf/arena.h"
#include "gtest/gtest.h"

#include "utl/progress_tracker.h"

#include "test_dir.h"

#ifdef PROTOBUF_LINKED
#include "google/protobuf/stubs/common.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::clog.rdbuf(std::cout.rdbuf());

  auto const progress_tracker = utl::activate_progress_tracker("test");
  auto const silencer = utl::global_progress_bars{true};
  fs::current_path(OSR_TEST_EXECUTION_DIR);

  ::testing::InitGoogleTest(&argc, argv);
  auto test_result = RUN_ALL_TESTS();

  google::protobuf::ShutdownProtobufLibrary();

  return test_result;
}