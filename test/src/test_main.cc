#include <filesystem>
#include <iostream>

#include "gtest/gtest.h"

#include "opentelemetry/trace/provider.h"

#include "utl/progress_tracker.h"

#include "motis/core/otel/tracer.h"

#include "test_dir.h"

#ifdef PROTOBUF_LINKED
#include "google/protobuf/stubs/common.h"
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::clog.rdbuf(std::cout.rdbuf());

  utl::get_active_progress_tracker_or_activate("test");

  fs::current_path(MOTIS_TEST_EXECUTION_DIR);
  std::cout << "executing tests in " << fs::current_path() << '\n';

  auto tracer_provider = opentelemetry::trace::Provider::GetTracerProvider();
  motis::motis_tracer = tracer_provider->GetTracer("motis-test");

  ::testing::InitGoogleTest(&argc, argv);
  auto test_result = RUN_ALL_TESTS();

#ifdef PROTOBUF_LINKED
  google::protobuf::ShutdownProtobufLibrary();
#endif

  return test_result;
}
