#include "motis/feed_cache.h"

#include <algorithm>
#include <cctype>

#include "boost/beast/core/buffers_to_string.hpp"

#include "utl/verify.h"

namespace motis {

namespace {

bool iequals(std::string_view const a, std::string_view const b) {
  return a.size() == b.size() &&
         std::equal(begin(a), end(a), begin(b), [](char const x, char const y) {
           return std::tolower(static_cast<unsigned char>(x)) ==
                  std::tolower(static_cast<unsigned char>(y));
         });
}

bool has_header(headers_t const& headers, std::string_view const name) {
  return std::any_of(begin(headers), end(headers),
                     [&](auto const& h) { return iequals(h.first, name); });
}

}  // namespace

std::int64_t feed_cache::now_unix() const {
  return std::chrono::duration_cast<std::chrono::seconds>(
             now_().time_since_epoch())
      .count();
}

feed_cache::entry const* feed_cache::get(std::string const& key) const {
  auto const it = entries_.find(key);
  return it == end(entries_) ? nullptr : &it->second;
}

void feed_cache::add_conditional_headers(std::string const& key,
                                         std::string_view const url,
                                         headers_t& headers) {
  auto const* cached = get(key);
  if (cached == nullptr || cached->url_ != url) {
    return;
  }
  if (!cached->etag_.empty() && !has_header(headers, "If-None-Match")) {
    headers.emplace("If-None-Match", cached->etag_);
  }
  if (!cached->last_modified_.empty() &&
      !has_header(headers, "If-Modified-Since")) {
    headers.emplace("If-Modified-Since", cached->last_modified_);
  }
}

feed_cache::result feed_cache::handle_response(std::string const& key,
                                               std::string_view const url,
                                               http_response const& res) {
  namespace http = boost::beast::http;

  auto const code = res.result_int();
  auto const now = now_unix();
  auto const header = [&](http::field const f) { return std::string{res[f]}; };

  if (code == 304) {
    auto const it = entries_.find(key);
    utl::verify(it != end(entries_),
                "HTTP 304 fetching {} without cached version", url);
    auto& cached = it->second;
    if (auto etag = header(http::field::etag); !etag.empty()) {
      cached.etag_ = std::move(etag);
    }
    if (auto last_modified = header(http::field::last_modified);
        !last_modified.empty()) {
      cached.last_modified_ = std::move(last_modified);
    }
    cached.confirmed_at_unix_ = now;
    return {.decoded_body_ =
                decode_http_body(cached.raw_body_, cached.content_encoding_),
            .status_ = status::kNotModified,
            .confirmed_at_ = clock::time_point{std::chrono::seconds{now}}};
  }

  utl::verify(code >= 200 && code < 300, "HTTP {} fetching {}", code, url);

  auto downloaded =
      entry{.url_ = std::string{url},
            .etag_ = header(http::field::etag),
            .last_modified_ = header(http::field::last_modified),
            .content_encoding_ = header(http::field::content_encoding),
            .downloaded_at_unix_ = now,
            .confirmed_at_unix_ = now,
            .raw_body_ = boost::beast::buffers_to_string(res.body().data())};

  auto decoded =
      decode_http_body(downloaded.raw_body_, downloaded.content_encoding_);
  entries_[key] = std::move(downloaded);
  return {.decoded_body_ = std::move(decoded),
          .status_ = status::kDownloaded,
          .confirmed_at_ = clock::time_point{std::chrono::seconds{now}}};
}

feed_cache::result feed_cache::get_stale(
    std::string const& key,
    std::chrono::seconds const max_age,
    std::string_view const download_error) {
  auto const* cached = get(key);
  utl::verify(cached != nullptr, "{} (no cached version available)",
              download_error);

  auto const age =
      std::chrono::seconds{now_unix() - cached->confirmed_at_unix_};
  utl::verify(age <= max_age, "{} (cached version too old: age={}s, max={}s)",
              download_error, age.count(), max_age.count());

  return {
      .decoded_body_ =
          decode_http_body(cached->raw_body_, cached->content_encoding_),
      .status_ = status::kStale,
      .confirmed_at_ =
          clock::time_point{std::chrono::seconds{cached->confirmed_at_unix_}},
      .download_error_ = std::string{download_error}};
}

boost::asio::awaitable<feed_cache::result> cached_http_GET(
    feed_cache* cache,
    std::string const& key,
    boost::urls::url const& url,
    headers_t headers,
    std::chrono::seconds const timeout,
    std::chrono::seconds const max_stale_age,
    std::optional<proxy> const& p,
    prepare_headers_fn const& prepare_headers) {
  auto const url_str = std::string{url.buffer()};
  auto download_error = std::string{};
  try {
    if (prepare_headers) {
      co_await prepare_headers(headers);
    }
    if (cache == nullptr) {
      auto const res = co_await http_GET(url, headers, timeout, p);
      utl::verify(res.result_int() >= 200 && res.result_int() < 300,
                  "HTTP {} fetching {}", res.result_int(), url_str);
      co_return feed_cache::result{.decoded_body_ = get_http_body(res),
                                   .status_ = feed_cache::status::kDownloaded,
                                   .confirmed_at_ = feed_cache::clock::now()};
    }
    cache->add_conditional_headers(key, url_str, headers);
    auto const res = co_await http_GET(url, headers, timeout, p);
    co_return cache->handle_response(key, url_str, res);
  } catch (std::exception const& e) {
    if (cache == nullptr || max_stale_age.count() <= 0) {
      throw;
    }
    download_error = e.what();
  }
  co_return cache->get_stale(key, max_stale_age, download_error);
}

}  // namespace motis
