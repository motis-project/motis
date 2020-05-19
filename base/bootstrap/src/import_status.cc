#include "motis/bootstrap/import_status.h"

#ifdef _MSC_VER
#include "windows.h"
#endif

#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>

#include "utl/to_vec.h"

namespace motis::bootstrap {

#ifdef _MSC_VER

constexpr auto const BAR = "\xDB";

void move(int x, int y) {
  auto hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (!hStdout) {
    return;
  }

  CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
  GetConsoleScreenBufferInfo(hStdout, &csbiInfo);

  COORD cursor;
  cursor.X = csbiInfo.dwCursorPosition.X + x;
  cursor.Y = csbiInfo.dwCursorPosition.Y + y;
  SetConsoleCursorPosition(hStdout, cursor);
}

void move_cursor_up(int lines) {
  if (lines != 0) {
    move(0, -lines);
  }
}

void set_vt100_mode() {
  auto hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  SetConsoleMode(hStdout,
                 ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
}

#else

constexpr auto const BAR = "■";

void set_vt100_mode() {}

void move_cursor_up(int lines) {
  if (lines != 0) {
    std::cout << "\x1b[" << lines << "A";
  }
}

#endif

void clear_line() { std::cout << "\x1b[K"; }

void import_status::set_progress_bounds(std::string const& name,  //
                                        float output_low, float output_high,
                                        float input_high) {
  auto s = status_[name];
  s.output_low_ = output_low;
  s.output_high_ = output_high;
  s.input_high_ = input_high;
  if (update(name, s)) {
    print();
  }
}

void import_status::update_progress(std::string const& name, float progress) {
  auto s = status_[name];
  s.status_ = import::Status_RUNNING;
  s.progress_ =
      std::clamp(std::round(s.output_low_ + (s.output_high_ - s.output_low_) *
                                                (progress / s.input_high_)),
                 0.F, 100.F);

  if (update(name, s)) {
    print();
  }
}

void import_status::report_error(std::string const& name,
                                 std::string const& what) {
  auto s = status_[name];
  s.status_ = import::Status_ERROR;
  s.error_ = what;
  if (update(name, s)) {
    print();
  }
}

void import_status::report_step(std::string const& name,
                                std::string const& step) {
  auto s = status_[name];
  s.status_ = import::Status_RUNNING;
  s.current_task_ = step;
  if (update(name, s)) {
    print();
  }
}

bool import_status::update(std::string const& task_name,
                           state const& new_state) {
  auto& old_state = status_[task_name];
  if (old_state != new_state) {
    old_state = new_state;
    return true;
  }
  return false;
}

bool import_status::update(motis::module::msg_ptr const& msg) {
  if (msg->get()->content_type() == MsgContent_StatusUpdate) {
    using import::StatusUpdate;
    auto const upd = motis_content(StatusUpdate, msg);
    return update(
        upd->name()->str(),
        {utl::to_vec(*upd->waiting_for(), [](auto&& e) { return e->str(); }),
         upd->status(), upd->progress(), 0.F, 100.F, 100.F, upd->error()->str(),
         upd->current_task()->str()});
  }
  return false;
}

void import_status::print() {
  if (silent_) {
    return;
  }
  set_vt100_mode();
  move_cursor_up(last_print_height_);
  for (auto const& [name, s] : status_) {
    clear_line();
    std::cout << std::setw(12) << std::setfill(' ') << std::right << name
              << ": ";
    switch (s.status_) {
      case import::Status_ERROR: std::cout << "ERROR: " << s.error_; break;
      case import::Status_WAITING:
        std::cout << "WAITING, dependencies: ";
        for (auto const& dep : s.dependencies_) {
          std::cout << dep << " ";
        }
        break;
      case import::Status_FINISHED: std::cout << "DONE"; break;
      case import::Status_RUNNING:
        std::cout << "[";
        constexpr auto const WIDTH = 55U;
        for (auto i = 0U; i < 55U; ++i) {
          auto const scaled = static_cast<int>(i * 100.0 / WIDTH);
          std::cout << (scaled <= s.progress_ ? BAR : " ");
        }
        std::cout << " ] " << std::setw(3) << s.progress_ << "%";
        if (!s.current_task_.empty()) {
          std::cout << " | " << s.current_task_;
        }
        break;
    }
    std::cout << "\n";
  }
  last_print_height_ = status_.size();
}

}  // namespace motis::bootstrap
