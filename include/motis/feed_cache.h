#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "boost/asio/awaitable.hpp"
#include "boost/url/url.hpp"

#include "motis/config.h"
#include "motis/http_req.h"
#include "motis/types.h"

namespace motis {

struct feed_cache {
  using clock = std::chrono::system_clock;

  struct entry {
    std::string url_;
    std::string etag_;
    std::string last_modified_;
    std::string content_encoding_;
    std::int64_t downloaded_at_unix_{0};
    std::int64_t confirmed_at_unix_{0};
    std::string raw_body_;
  };

  enum class status : std::uint8_t { kDownloaded, kNotModified, kStale };

  struct result {
    std::string decoded_body_;
    status status_{status::kDownloaded};
    clock::time_point confirmed_at_{};
    std::string download_error_{};
  };

  entry const* get(std::string const& key) const;

  void add_conditional_headers(std::string const& key,
                               std::string_view url,
                               headers_t& headers);

  result handle_response(std::string const& key,
                         std::string_view url,
                         http_response const&);

  result get_stale(std::string const& key,
                   std::chrono::seconds max_age,
                   std::string_view download_error);

  std::function<clock::time_point()> now_{[]() { return clock::now(); }};

private:
  std::int64_t now_unix() const;

  hash_map<std::string, entry> entries_;
};

using prepare_headers_fn =
    std::function<boost::asio::awaitable<void>(headers_t&)>;

boost::asio::awaitable<feed_cache::result> cached_http_GET(
    feed_cache* cache,
    std::string const& key,
    boost::urls::url const& url,
    headers_t headers,
    std::chrono::seconds timeout,
    std::chrono::seconds max_stale_age,
    std::optional<proxy> const& = std::nullopt,
    prepare_headers_fn const& prepare_headers = {});

}  // namespace motis
