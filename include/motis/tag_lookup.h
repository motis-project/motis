#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "cista/memory_holder.h"

#include "nigiri/types.h"

#include "motis/fwd.h"

namespace motis {

struct tag_lookup {
  void add(nigiri::source_idx_t, std::string_view str);

  nigiri::source_idx_t get_src(std::string_view tag) const;
  std::string_view get_tag(nigiri::source_idx_t) const;
  std::string id(nigiri::timetable const&, nigiri::location_idx_t) const;
  std::string id(nigiri::timetable const&,
                 nigiri::rt::run_stop,
                 nigiri::event_type) const;
  nigiri::location_idx_t get_location(nigiri::timetable const&,
                                      std::string_view) const;
  std::pair<nigiri::rt::run, nigiri::trip_idx_t> get_trip(
      nigiri::timetable const&, std::string_view) const;

  friend std::ostream& operator<<(std::ostream&, tag_lookup const&);
  void write(std::filesystem::path const&) const;
  static cista::wrapped<tag_lookup> read(std::filesystem::path const&);

  nigiri::vecvec<nigiri::source_idx_t, char, std::uint32_t> src_to_tag_;
  nigiri::hash_map<nigiri::string, nigiri::source_idx_t> tag_to_src_;
};

}  // namespace motis