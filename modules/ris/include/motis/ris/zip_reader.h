#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "utl/parser/buffer.h"

namespace motis::ris {

struct zip_reader {
  zip_reader(char const* ptr, size_t size);
  explicit zip_reader(char const* path);

  ~zip_reader();

  zip_reader(zip_reader&&) = default;
  zip_reader& operator=(zip_reader&&) = default;

  zip_reader(zip_reader const&) = delete;
  zip_reader& operator=(zip_reader const&) = delete;

  std::optional<std::string_view> read() const;
  float progress() const;
  std::string_view current_file_name() const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::ris
