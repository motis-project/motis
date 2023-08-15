#pragma once

#include <string>
#include <string_view>

#include "nigiri/types.h"

namespace motis::nigiri {

struct tag_lookup {
  void add(::nigiri::source_idx_t, std::string_view str);

  ::nigiri::source_idx_t get_src(std::string_view tag) const;
  std::string_view get_tag(::nigiri::source_idx_t const src) const;
  std::string_view get_tag_clean(::nigiri::source_idx_t const src) const;

  friend std::ostream& operator<<(std::ostream& out, tag_lookup const& tags);

  ::nigiri::vecvec<::nigiri::source_idx_t, char, std::uint32_t> src_to_tag_;
  ::nigiri::hash_map<std::string, ::nigiri::source_idx_t> tag_to_src_;
};

}  // namespace motis::nigiri