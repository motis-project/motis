#pragma once

#include <memory>
#include <string>

#include "motis/module/context/motis_http_req.h"
#include "motis/nigiri/tag_lookup.h"

namespace motis::nigiri {

struct gtfsrt {
  // Config format: tag|url|auth
  // Example 1: nl|http://gtfs.ovapi.nl/nl/tripUpdates.pb|my_api_key
  // Example 2: nl|http://gtfs.ovapi.nl/nl/tripUpdates.pb
  gtfsrt(tag_lookup const&, std::string_view config);
  gtfsrt(gtfsrt&&) noexcept;
  gtfsrt& operator=(gtfsrt&&) noexcept;
  ~gtfsrt();

  motis::module::http_future_t fetch() const;
  ::nigiri::source_idx_t src() const;

  struct impl;
  std::unique_ptr<impl> impl_;
};

}  // namespace motis::nigiri