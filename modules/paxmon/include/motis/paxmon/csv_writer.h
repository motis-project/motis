#pragma once

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

namespace motis::paxmon {

// RFC 4180

static constexpr struct end_row_t {
} end_row{};

struct basic_csv_writer {
  virtual ~basic_csv_writer() = default;

  explicit operator bool() { return static_cast<bool>(stream()); }
  bool operator!() { return !stream(); }

  virtual bool is_open() const = 0;

  template <typename T>
  basic_csv_writer& operator<<(T val) {
    if (is_open()) {
      start_col();
      stream() << val;
    }
    return *this;
  }

  basic_csv_writer& operator<<(std::string const& str) {
    if (is_open()) {
      start_col();
      stream() << std::quoted(str, '"', '"');
    }
    return *this;
  }

  basic_csv_writer& operator<<(std::string_view str) {
    if (is_open()) {
      start_col();
      stream() << std::quoted(str, '"', '"');
    }
    return *this;
  }

  basic_csv_writer& operator<<(char const* str) {
    if (is_open()) {
      start_col();
      stream() << std::quoted(str, '"', '"');
    }
    return *this;
  }

  basic_csv_writer& operator<<(end_row_t const&) {
    if (is_open()) {
      end_row();
    }
    return *this;
  }

  void end_row() {
    if (is_open()) {
      stream() << "\n";
      first_col_ = true;
    }
  }

  void flush() {
    if (is_open()) {
      stream().flush();
    }
  }

  void enable_exceptions() {
    stream().exceptions(std::ofstream::failbit | std::ofstream::badbit);
  }

  virtual std::ostream& stream() = 0;

private:
  void start_col() {
    if (first_col_) {
      first_col_ = false;
    } else {
      stream() << separator_;
    }
  }

  bool first_col_{true};

public:
  char separator_{','};
};

struct file_csv_writer : public basic_csv_writer {
  explicit file_csv_writer(std::string const& filename) : basic_csv_writer{} {
    if (!filename.empty()) {
      ofs_.open(filename);
    }
  }

  bool is_open() const override { return ofs_.is_open(); }
  std::ostream& stream() override { return ofs_; }

private:
  std::ofstream ofs_;
};

struct string_csv_writer : public basic_csv_writer {
  bool is_open() const override { return true; }
  std::ostream& stream() override { return stream_; }
  std::string str() const { return stream_.str(); }

private:
  std::stringstream stream_;
};

}  // namespace motis::paxmon
