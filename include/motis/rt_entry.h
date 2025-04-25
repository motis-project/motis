#pragma once

#include <string_view>

#include "cista/hashing.h"

#include "motis/types.h"

namespace motis {

struct rt_entry {

  struct gtfsrt {
    std::string_view url_;
    headers_t const* headers_;
  };

  struct vdvaus {
    std::string_view url_;
    std::string const& server_name_;
    std::string const& client_name_;
    unsigned hysteresis_;
  };

  std::variant<gtfsrt, vdvaus> operator()() const;

  bool operator==(rt_entry const&) const = default;
  cista::hash_t hash() const noexcept {
    return cista::build_hash(protocol_, url_, headers_, client_name_,
                             server_name_, hysteresis_);
  }

  enum struct protocol { gtfsrt, vdvaus };
  std::optional<protocol> protocol_;

  std::string url_;

  // GTFS-RT
  std::optional<headers_t> headers_{};

  // VDV AUS
  std::optional<std::string> server_name_;
  std::optional<std::string> client_name_;
  std::optional<unsigned> hysteresis_;
};

}  // namespace motis