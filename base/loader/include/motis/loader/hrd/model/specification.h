#pragma once

#include <vector>

#include "utl/parser/cstr.h"

namespace motis::loader::hrd {

constexpr char const* UNKNOWN_FILE = "unknown_file";
constexpr int BEFORE_FIRST_LINE = -1;

struct specification {
  specification()
      : filename_(UNKNOWN_FILE),
        line_number_from_(BEFORE_FIRST_LINE),
        line_number_to_(BEFORE_FIRST_LINE),
        internal_service_(nullptr, 0) {}

  bool is_empty() const;

  bool valid() const;

  bool ignore() const;

  void reset();

  bool read_line(utl::cstr line, char const* filename, int line_number);

  char const* filename_;
  int line_number_from_;
  int line_number_to_;
  utl::cstr internal_service_;
  std::vector<utl::cstr> traffic_days_;
  std::vector<utl::cstr> categories_;
  std::vector<utl::cstr> line_information_;
  std::vector<utl::cstr> attributes_;
  std::vector<utl::cstr> directions_;
  std::vector<utl::cstr> stops_;
};

}  // namespace motis::loader::hrd
