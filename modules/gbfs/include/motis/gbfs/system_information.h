#pragma once

#include <string>
#include <string_view>

namespace motis::gbfs {

struct system_information {
  std::string name_;
  std::string name_short_;
  std::string operator_;
  std::string url_;
  std::string purchase_url_;
  std::string mail_;
};

system_information read_system_information(std::string_view);

}  // namespace motis::gbfs
