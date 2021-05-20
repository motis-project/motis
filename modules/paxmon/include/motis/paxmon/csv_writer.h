#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <string_view>

namespace motis::paxmon {

// RFC 4180

static constexpr struct end_row_t {
} end_row{};

struct csv_writer {
  explicit csv_writer(std::string const& filename) : ofs_{}, first_col_{true} {
    if (!filename.empty()) {
      ofs_.open(filename);
    }
  }

  explicit operator bool() const { return static_cast<bool>(ofs_); }
  bool operator!() const { return !ofs_; }

  inline bool is_open() const { return ofs_.is_open(); }

  template <typename T>
  csv_writer& operator<<(T val) {
    if (is_open()) {
      start_col();
      ofs_ << val;
    }
    return *this;
  }

  csv_writer& operator<<(std::string const& str) {
    if (is_open()) {
      start_col();
      ofs_ << std::quoted(str, '"', '"');
    }
    return *this;
  }

  csv_writer& operator<<(std::string_view str) {
    if (is_open()) {
      start_col();
      ofs_ << std::quoted(str, '"', '"');
    }
    return *this;
  }

  csv_writer& operator<<(char const* str) {
    if (is_open()) {
      start_col();
      ofs_ << std::quoted(str, '"', '"');
    }
    return *this;
  }

  csv_writer& operator<<(end_row_t const&) {
    if (is_open()) {
      end_row();
    }
    return *this;
  }

  void end_row() {
    if (is_open()) {
      ofs_ << "\n";
      first_col_ = true;
    }
  }

  void flush() {
    if (is_open()) {
      ofs_.flush();
    }
  }

  void enable_exceptions() {
    ofs_.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  }

  std::ofstream& stream() { return ofs_; }

private:
  void start_col() {
    if (first_col_) {
      first_col_ = false;
    } else {
      ofs_ << ",";
    }
  }

  std::ofstream ofs_;
  bool first_col_;
};

}  // namespace motis::paxmon
