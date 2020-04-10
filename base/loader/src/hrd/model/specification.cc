#include "motis/loader/hrd/model/specification.h"

#include <cctype>

#include "motis/loader/parser_error.h"

using namespace utl;

namespace motis::loader::hrd {

bool specification::is_empty() const {
  return internal_service_.str == nullptr || internal_service_.len == 0;
}

bool specification::valid() const {
  return ignore() || (!categories_.empty() && stops_.size() >= 2 &&
                      !traffic_days_.empty() && !is_empty());
}

bool specification::ignore() const {
  return !is_empty() && !internal_service_.starts_with("*Z");
}

void specification::reset() {
  internal_service_ = cstr(nullptr, 0);
  traffic_days_.clear();
  categories_.clear();
  line_information_.clear();
  attributes_.clear();
  directions_.clear();
  stops_.clear();
}

bool specification::read_line(cstr line, char const* filename,
                              int line_number) {
  if (line.len == 0) {
    return false;
  }

  if (std::isdigit(line[0]) != 0) {
    stops_.push_back(line);
    return false;
  }

  if (line.len < 2 || line[0] != '*') {
    throw parser_error(filename, line_number);
  }

  // ignore *I, *GR, *SH, *T, *KW, *KWZ
  bool potential_kurswagen = false;
  switch (line[1]) {
    case 'K': potential_kurswagen = true;
    /* no break */
    case 'Z':
    case 'T':
      if (potential_kurswagen && line.len > 3 && line[3] == 'Z') {
        // ignore KWZ line
      } else if (is_empty()) {
        filename_ = filename;
        line_number_from_ = line_number;
        internal_service_ = line;
      } else {
        return true;
      }
      break;
    case 'A':
      if (line.starts_with("*A VE")) {
        traffic_days_.push_back(line);
      } else {  // *A based on HRD format version 5.00.8
        attributes_.push_back(line);
      }
      break;
    case 'G':
      if (!line.starts_with("*GR")) {
        categories_.push_back(line);
      }
      break;
    case 'L': line_information_.push_back(line); break;
    case 'R': directions_.push_back(line); break;
  }

  return false;
}

}  // namespace motis::loader::hrd
