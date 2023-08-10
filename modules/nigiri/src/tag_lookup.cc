#include "motis/nigiri/tag_lookup.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

namespace n = ::nigiri;

namespace motis::nigiri {

void tag_lookup::add(n::source_idx_t const src, std::string_view str) {
  utl::verify(tag_to_src_.size() == to_idx(src), "invalid tag");
  tag_to_src_.emplace(std::string{str}, src);
  src_to_tag_.emplace_back(str);
}

n::source_idx_t tag_lookup::get_src(std::string_view tag) const {
  auto const it = tag_to_src_.find(tag);
  return it == end(tag_to_src_) ? n::source_idx_t::invalid() : it->second;
}

std::string_view tag_lookup::get_tag(n::source_idx_t const src) const {
  return src == n::source_idx_t::invalid() ? "" : src_to_tag_.at(src).view();
}

std::string_view tag_lookup::get_tag_clean(n::source_idx_t const src) const {
  auto const tag = get_tag(src);
  return tag.empty() ? tag : tag.substr(0, tag.size() - 1);
}

std::ostream& operator<<(std::ostream& out, tag_lookup const& tags) {
  auto first = true;
  for (auto const [src, tag] : utl::enumerate(tags.src_to_tag_)) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << src << "=" << tag.view();
  }
  return out;
}

}  // namespace motis::nigiri