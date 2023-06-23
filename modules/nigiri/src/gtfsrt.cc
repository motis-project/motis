#include "motis/nigiri/gtfsrt.h"

#include "utl/parser/split.h"

#include "net/http/client/request.h"

#include "motis/nigiri/location.h"

namespace mm = motis::module;
namespace n = nigiri;

namespace motis::nigiri {

struct gtfsrt::impl {
  impl(net::http::client::request req, n::source_idx_t const src)
      : req_{std::move(req)}, src_{src} {}
  net::http::client::request req_;
  n::source_idx_t src_;
};

gtfsrt::gtfsrt(tag_lookup const& tags, std::string_view config) {
  auto const [tag, url, auth] =
      utl::split<'|', utl::cstr, utl::cstr, utl::cstr>(config);
  auto const src = tags.get_src(tag.to_str() + "_");
  utl::verify(
      src != n::source_idx_t::invalid(),
      "nigiri GTFS-RT tag {} not found as static timetable (known tags: {})",
      tag.view(), tags);
  auto req = net::http::client::request{url.to_str()};
  if (!auth.empty()) {
    req.headers.emplace("Authorization", auth.to_str());
  }
  impl_ = std::make_unique<impl>(std::move(req), src);
}

gtfsrt::gtfsrt(gtfsrt&&) noexcept = default;
gtfsrt& gtfsrt::operator=(gtfsrt&&) noexcept = default;

gtfsrt::~gtfsrt() = default;

mm::http_future_t gtfsrt::fetch() const { return motis_http(impl_->req_); }

n::source_idx_t gtfsrt::src() const { return impl_->src_; }

}  // namespace motis::nigiri