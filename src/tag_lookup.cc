#include "motis/tag_lookup.h"

#include "fmt/core.h"

#include "cista/mmap.h"
#include "cista/serialization.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {

constexpr auto const kMode =
    cista::mode::WITH_INTEGRITY | cista::mode::WITH_STATIC_VERSION;

std::pair<std::string_view, std::string_view> split_tag_and_location_id(
    std::string_view station_id) {
  auto const first_underscore_pos = station_id.find('_');
  return first_underscore_pos != std::string_view::npos
             ? std::pair{station_id.substr(0, first_underscore_pos),
                         station_id.substr(first_underscore_pos + 1U)}
             : std::pair{std::string_view{}, station_id};
}

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

std::string tag_lookup::id(nigiri::timetable const& tt,
                           nigiri::location_idx_t const l) const {
  auto const src = tt.locations_.src_.at(l);
  auto const id = tt.locations_.ids_.at(l).view();
  return src == n::source_idx_t::invalid()
             ? std::string{id}
             : fmt::format("{}_{}", get_tag(src), id);
}

nigiri::location_idx_t tag_lookup::get(nigiri::timetable const& tt,
                                       std::string_view s) const {
  auto const [tag, id] = split_tag_and_location_id(s);
  auto const src = get_src(tag);
  try {
    return tt.locations_.location_id_to_idx_.at({{id}, src});
  } catch (...) {
    throw utl::fail(
        R"(could not find timetable location "{}", tag="{}", id="{}", src={})",
        s, tag, id, static_cast<int>(to_idx(src)));
  }
}

void tag_lookup::write(std::filesystem::path const& p) const {
  auto mmap = cista::mmap{p.string().c_str(), cista::mmap::protection::WRITE};
  auto writer = cista::buf<cista::mmap>(std::move(mmap));
  cista::serialize<kMode>(writer, *this);
}

cista::wrapped<tag_lookup> tag_lookup::read(std::filesystem::path const& p) {
  auto b = cista::file{p.generic_string().c_str(), "r"}.content();
  auto const ptr = cista::deserialize<tag_lookup, kMode>(b);
  auto mem = cista::memory_holder{std::move(b)};
  return cista::wrapped{std::move(mem), ptr};
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

}  // namespace motis