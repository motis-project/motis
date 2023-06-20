#include "motis/nigiri/gtfsrt.h"

#include "utl/parser/split.h"

#include "net/http/client/request.h"

#include "motis/nigiri/location.h"

namespace mm = motis::module;
namespace n = nigiri;

namespace motis::nigiri {

struct gtfsrt::impl {
  net::http::client::request req_;
  n::source_idx_t src_;
};

gtfsrt::gtfsrt(tag_lookup const& tags, std::string const& config) {
  auto const [tag, url, auth] =
      utl::split<'|', utl::cstr, utl::cstr, utl::cstr>(config);
  auto req = net::http::client::request{url.to_str()};
  if (!auth.empty()) {
    req.headers.emplace("Authorization", auth.to_str());
  }
  impl_ = std::make_unique<impl>(std::move(req), tags.get_src(tag.view()));
}

gtfsrt::~gtfsrt() = default;

mm::http_future_t gtfsrt::fetch() { return motis_http(impl_->req_); }

n::source_idx_t gtfsrt::src() const { return impl_->src_; }

}  // namespace motis::nigiri