#include <iostream>

#include "fmt/printf.h"

using namespace std::string_view_literals;

int main(int ac, char** av) {
  if (ac > 1 && av[1] == "--help"sv) {
    fmt::println(
        "calibrate a model based on connection sets\n\n"
        "Usage:\n"
        "odm_calibrate requirements.json\n");
    return 0;
  }
}