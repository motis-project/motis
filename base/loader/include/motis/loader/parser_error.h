#pragma once

#include <stdexcept>

namespace motis::loader {

struct parser_error : public std::exception {
  parser_error(char const* filename, int line_number)
      : filename_copy_(filename),
        filename_(filename_copy_.c_str()),
        line_number_(line_number) {}

  char const* what() const noexcept override { return "parser_error"; }

  void print_what() const noexcept {
    printf("%s:%s:%d\n", what(), filename_, line_number_);
  }

  std::string filename_copy_;
  char const* filename_;
  int line_number_;
};

}  // namespace motis::loader
