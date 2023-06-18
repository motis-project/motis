#include "motis/nigiri/gtfsrt.h"

#include "utl/parser/split.h"

#include "net/http/client/request.h"

#include "motis/nigiri/location.h"

namespace n = ::nigiri;

namespace motis::nigiri {

struct gtfsrt::impl {
  net::http::client::request req_;
  n::source_idx_t src_;
};

gtfsrt::gtfsrt(motis::nigiri::tag_map_t const& tags, std::string const& config)
    : impl_{std::make_unique<impl>()} {
  auto const [tag, url, auth] =
      utl::split<'|', utl::cstr, utl::cstr, utl::cstr>(config);
  auto req = net::http::client::request{url.to_str()};
  if (!auth.empty()) {
    req.headers.emplace("Authorization", auth.to_str());
  }
}

}  // namespace motis::nigiri