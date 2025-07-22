#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "cista/memory_holder.h"

#include "nigiri/types.h"

#include "motis/fwd.h"

namespace motis {

template <typename T = std::string_view>
struct trip_id {
  T start_date_;
  T start_time_;
  T tag_;
  T trip_id_;
};

trip_id<std::string_view> split_trip_id(std::string_view);

struct tag_lookup {
  void add(nigiri::source_idx_t, std::string_view str);

  nigiri::source_idx_t get_src(std::string_view tag) const;
  std::string_view get_tag(nigiri::source_idx_t) const;
  std::string id(nigiri::timetable const&, nigiri::location_idx_t) const;
  std::string id(nigiri::timetable const&,
                 nigiri::rt::run_stop,
                 nigiri::event_type) const;

  trip_id<std::string> id_fragments(nigiri::timetable const&,
                                    nigiri::rt::run_stop,
                                    nigiri::event_type const) const;

  nigiri::location_idx_t get_location(nigiri::timetable const&,
                                      std::string_view) const;
  std::pair<nigiri::rt::run, nigiri::trip_idx_t> get_trip(
      nigiri::timetable const&,
      nigiri::rt_timetable const*,
      std::string_view) const;

  friend std::ostream& operator<<(std::ostream&, tag_lookup const&);
  void write(std::filesystem::path const&) const;
  static cista::wrapped<tag_lookup> read(std::filesystem::path const&);

  nigiri::vecvec<nigiri::source_idx_t, char, std::uint32_t> src_to_tag_;
  nigiri::hash_map<nigiri::string, nigiri::source_idx_t> tag_to_src_;
};

}  // namespace motis