#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "motis/module/clog_redirect.h"

namespace mm = motis::module;
namespace t = testing;

struct progress_listener_mock : public mm::progress_listener {
  MOCK_METHOD(void, set_progress_bounds,
              (std::string const& name, double output_low, double output_high,
               double input_high),
              (override));
  MOCK_METHOD(void, update_progress, (std::string const& name, double progress),
              (override));
  MOCK_METHOD(void, report_error,
              (std::string const& name, std::string const& what), (override));
  MOCK_METHOD(void, report_step,
              (std::string const& name, std::string const& task), (override));
};

#ifdef _MSC_VER
constexpr auto const kDevNull = "nul";
#else
constexpr auto const kDevNull = "/dev/null";
#endif

TEST(clog_redirect_test, passthrough) {
  t::StrictMock<progress_listener_mock> pl;

  mm::clog_redirect::set_enabled(true);
  mm::clog_redirect redirect{pl, "test", kDevNull};
  std::clog << "just some boring log message\n";
}

TEST(clog_redirect_test, report_step) {
  t::StrictMock<progress_listener_mock> pl;
  EXPECT_CALL(pl, report_step("test", "new status"));

  mm::clog_redirect::set_enabled(true);
  mm::clog_redirect redirect{pl, "test", kDevNull};
  std::clog << '\0' << 'S' << "new status" << '\0';
}

TEST(clog_redirect_test, report_error) {
  t::StrictMock<progress_listener_mock> pl;
  EXPECT_CALL(pl, report_error("test", "have error"));

  mm::clog_redirect::set_enabled(true);
  mm::clog_redirect redirect{pl, "test", kDevNull};
  std::clog << '\0' << 'E' << "have error" << '\0';
}

TEST(clog_redirect_test, update_progress) {
  t::StrictMock<progress_listener_mock> pl;
  {
    testing::InSequence seq;
    EXPECT_CALL(pl, update_progress("test", 0));
    EXPECT_CALL(pl, update_progress("test", 37.5));
    EXPECT_CALL(pl, update_progress("test", 50));
    EXPECT_CALL(pl, update_progress("test", 100));
    EXPECT_CALL(pl, update_progress("test", 8888));
    EXPECT_CALL(pl, update_progress("test", -1));
  }

  mm::clog_redirect::set_enabled(true);
  mm::clog_redirect redirect{pl, "test", kDevNull};
  std::clog << '\0' << "0" << '\0';
  std::clog << '\0' << "37.5" << '\0';
  std::clog << '\0' << "50" << '\0';
  std::clog << '\0' << "100" << '\0';
  std::clog << '\0' << "8888" << '\0';
  std::clog << '\0' << "-1" << '\0';
}

TEST(clog_redirect_test, set_progress_bounds) {
  t::StrictMock<progress_listener_mock> pl;
  {
    testing::InSequence seq;
    EXPECT_CALL(pl, set_progress_bounds("test", 0, 100, 100));
    EXPECT_CALL(pl, set_progress_bounds("test", 0, 40, 1.0));
    EXPECT_CALL(pl, set_progress_bounds("test", 40, 80, 37));
    EXPECT_CALL(pl, set_progress_bounds("test", 80, 100, 100));
  }

  mm::clog_redirect::set_enabled(true);
  mm::clog_redirect redirect{pl, "test", kDevNull};
  std::clog << '\0' << 'B' << "0 100 100" << '\0';
  std::clog << '\0' << 'B' << "0 40 1.0" << '\0';
  std::clog << '\0' << 'B' << "40 80 37" << '\0';
  std::clog << '\0' << 'B' << "80 100" << '\0';
}
