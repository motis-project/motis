#include "gtest/gtest.h"

#include <chrono>
#include <optional>
#include <sstream>
#include <string>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/io_context.hpp"
#include "boost/iostreams/copy.hpp"
#include "boost/iostreams/filter/gzip.hpp"
#include "boost/iostreams/filtering_stream.hpp"

#include "motis/feed_cache.h"

#include "./http_test_server.h"

namespace http = boost::beast::http;
using namespace motis;
using motis::test::http_test_server;
using namespace std::chrono_literals;

namespace {

constexpr auto const kKey = "feed";
constexpr auto const kLastModified = "Wed, 21 Oct 2026 07:28:00 GMT";

std::string gzip(std::string const& in) {
  auto out = std::stringstream{};
  auto os = boost::iostreams::filtering_ostream{};
  os.push(boost::iostreams::gzip_compressor{});
  os.push(out);
  os << in;
  os.reset();
  return out.str();
}

struct conditional_feed_server {
  explicit conditional_feed_server(boost::asio::io_context& ioc)
      : srv_{ioc, [this](auto const& req) { return handle(req); }} {}

  http_test_server::response_t handle(http_test_server::request_t const& req) {
    if (status_ != http::status::ok) {
      return http_test_server::make_response(req, status_, "error");
    }
    auto const etag = fmt::format("\"v{}\"", version_);
    if (support_conditional_ && req[http::field::if_none_match] == etag) {
      return http_test_server::make_response(req, http::status::not_modified,
                                             "", {{http::field::etag, etag}});
    }
    auto headers = std::vector<std::pair<http::field, std::string>>{
        {http::field::etag, etag}, {http::field::last_modified, kLastModified}};
    if (gzip_) {
      headers.emplace_back(http::field::content_encoding, "gzip");
    }
    return http_test_server::make_response(
        req, http::status::ok, gzip_ ? gzip(body()) : body(), headers);
  }

  std::string body() const { return fmt::format("content v{}", version_); }

  http_test_server::request_t const& last_request() const {
    return srv_.requests_.back();
  }

  http_test_server srv_;
  unsigned version_{1U};
  http::status status_{http::status::ok};
  bool gzip_{false};
  bool support_conditional_{true};
};

template <typename Fn>
void run_until_client_done(boost::asio::io_context& ioc, Fn&& fn) {
  auto ex = std::exception_ptr{};
  boost::asio::co_spawn(
      ioc,
      [&]() -> boost::asio::awaitable<void> {
        try {
          co_await fn();
        } catch (...) {
          ex = std::current_exception();
        }
        ioc.stop();
      },
      boost::asio::detached);
  ioc.run();
  ioc.restart();
  if (ex) {
    std::rethrow_exception(ex);
  }
}

struct fixture {
  fixture() = default;
  fixture(fixture const&) = delete;
  fixture& operator=(fixture const&) = delete;
  ~fixture() {
    server_.srv_.stop();
    ioc_.restart();
    ioc_.run();
  }

  feed_cache::result fetch(feed_cache* cache,
                           std::chrono::seconds const max_age = 3600s,
                           std::optional<std::string> url = std::nullopt,
                           headers_t headers = {},
                           prepare_headers_fn const& prepare_headers = {}) {
    auto ret = feed_cache::result{};
    run_until_client_done(ioc_, [&]() -> boost::asio::awaitable<void> {
      ret = co_await cached_http_GET(
          cache, kKey, boost::urls::url{url.value_or(server_.srv_.url("rt"))},
          std::move(headers), 5s, max_age, std::nullopt, prepare_headers);
    });
    return ret;
  }

  boost::asio::io_context ioc_;
  conditional_feed_server server_{ioc_};
};

}  // namespace

TEST(motis, feed_cache_sends_conditional_request_and_uses_cache_on_304) {
  auto f = fixture{};
  auto cache = feed_cache{};

  auto const r1 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kDownloaded, r1.status_);
  EXPECT_EQ("content v1", r1.decoded_body_);
  EXPECT_TRUE(f.server_.last_request()[http::field::if_none_match].empty());
  EXPECT_TRUE(f.server_.last_request()[http::field::if_modified_since].empty());

  auto const r2 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kNotModified, r2.status_);
  EXPECT_EQ("content v1", r2.decoded_body_);
  EXPECT_EQ("\"v1\"", f.server_.last_request()[http::field::if_none_match]);
  EXPECT_EQ(kLastModified,
            f.server_.last_request()[http::field::if_modified_since]);

  f.server_.version_ = 2U;
  auto const r3 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kDownloaded, r3.status_);
  EXPECT_EQ("content v2", r3.decoded_body_);
  EXPECT_EQ("\"v1\"", f.server_.last_request()[http::field::if_none_match]);

  auto const r4 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kNotModified, r4.status_);
  EXPECT_EQ("content v2", r4.decoded_body_);
  EXPECT_EQ("\"v2\"", f.server_.last_request()[http::field::if_none_match]);
}

TEST(motis, feed_cache_server_without_conditional_support) {
  auto f = fixture{};
  f.server_.support_conditional_ = false;
  auto cache = feed_cache{};

  EXPECT_EQ(feed_cache::status::kDownloaded, f.fetch(&cache).status_);
  auto const r = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kDownloaded, r.status_);
  EXPECT_EQ("content v1", r.decoded_body_);
}

TEST(motis, feed_cache_does_not_override_user_headers) {
  auto f = fixture{};
  auto cache = feed_cache{};
  f.fetch(&cache);
  auto const r = f.fetch(&cache, 3600s, std::nullopt,
                         headers_t{{"if-none-match", "\"custom\""}});
  EXPECT_EQ(feed_cache::status::kDownloaded, r.status_);
  EXPECT_EQ("\"custom\"", f.server_.last_request()[http::field::if_none_match]);
}

TEST(motis, feed_cache_stale_on_http_error) {
  auto f = fixture{};
  auto cache = feed_cache{};
  f.fetch(&cache);

  f.server_.status_ = http::status::internal_server_error;
  auto const r = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kStale, r.status_);
  EXPECT_EQ("content v1", r.decoded_body_);
  EXPECT_NE(std::string::npos, r.download_error_.find("HTTP 500"));

  // recovered
  f.server_.status_ = http::status::ok;
  EXPECT_EQ(feed_cache::status::kNotModified, f.fetch(&cache).status_);
}

TEST(motis, feed_cache_stale_on_connection_error) {
  auto f = fixture{};
  auto cache = feed_cache{};
  f.fetch(&cache);

  // connection refused
  f.server_.srv_.stop();
  auto const url = f.server_.srv_.url("rt");
  auto ioc = boost::asio::io_context{};
  auto ret = feed_cache::result{};
  boost::asio::co_spawn(
      ioc,
      [&]() -> boost::asio::awaitable<void> {
        ret = co_await cached_http_GET(&cache, kKey, boost::urls::url{url}, {},
                                       5s, 3600s);
      },
      boost::asio::detached);
  ioc.run();

  EXPECT_EQ(feed_cache::status::kStale, ret.status_);
  EXPECT_EQ("content v1", ret.decoded_body_);
  EXPECT_FALSE(ret.download_error_.empty());
}

TEST(motis, feed_cache_not_stale_if_too_old_or_disabled_or_missing) {
  auto f = fixture{};
  auto cache = feed_cache{};

  f.server_.status_ = http::status::service_unavailable;
  EXPECT_THROW(f.fetch(&cache), std::exception);

  f.server_.status_ = http::status::ok;
  f.fetch(&cache);
  f.server_.status_ = http::status::service_unavailable;

  EXPECT_THROW(f.fetch(&cache, 0s), std::exception);

  auto const now = feed_cache::clock::now();
  cache.now_ = [&]() { return now + 2h; };
  EXPECT_THROW(f.fetch(&cache, 1h), std::exception);
  EXPECT_EQ(feed_cache::status::kStale, f.fetch(&cache, 3h).status_);
}

TEST(motis, feed_cache_304_refreshes_stale_age) {
  auto f = fixture{};
  auto cache = feed_cache{};
  auto now = feed_cache::clock::now();
  cache.now_ = [&]() { return now; };

  f.fetch(&cache);
  now += 50min;
  EXPECT_EQ(feed_cache::status::kNotModified, f.fetch(&cache).status_);
  now += 50min;  // 50min since last 304
  f.server_.status_ = http::status::bad_gateway;
  EXPECT_EQ(feed_cache::status::kStale, f.fetch(&cache, 1h).status_);
}

TEST(motis, feed_cache_without_cache_throws_on_http_error) {
  auto f = fixture{};
  EXPECT_EQ("content v1", f.fetch(nullptr).decoded_body_);
  f.server_.status_ = http::status::not_found;
  EXPECT_THROW(f.fetch(nullptr), std::exception);
}

TEST(motis, feed_cache_url_change_disables_conditional_request) {
  auto f = fixture{};
  auto cache = feed_cache{};
  f.fetch(&cache);

  auto const r = f.fetch(&cache, 3600s, f.server_.srv_.url("rt?token=2"));
  EXPECT_EQ(feed_cache::status::kDownloaded, r.status_);
  EXPECT_TRUE(f.server_.last_request()[http::field::if_none_match].empty());
}

TEST(motis, feed_cache_gzip_body_is_decoded_on_every_use) {
  auto f = fixture{};
  f.server_.gzip_ = true;
  auto cache = feed_cache{};

  auto const r1 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kDownloaded, r1.status_);
  EXPECT_EQ("content v1", r1.decoded_body_);

  auto const* e = cache.get(kKey);
  ASSERT_NE(nullptr, e);
  EXPECT_EQ("gzip", e->content_encoding_);
  ASSERT_GE(e->raw_body_.size(), 2U);
  EXPECT_EQ('\x1f', e->raw_body_[0]);
  EXPECT_EQ('\x8b', e->raw_body_[1]);

  auto const r2 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kNotModified, r2.status_);
  EXPECT_EQ("content v1", r2.decoded_body_);

  f.server_.status_ = http::status::internal_server_error;
  auto const r3 = f.fetch(&cache);
  EXPECT_EQ(feed_cache::status::kStale, r3.status_);
  EXPECT_EQ("content v1", r3.decoded_body_);
}

TEST(motis, feed_cache_prepare_headers_error_uses_stale) {
  auto f = fixture{};
  auto cache = feed_cache{};
  auto const add_token =
      [](headers_t& headers) -> boost::asio::awaitable<void> {
    headers["Authorization"] = "Bearer token";
    co_return;
  };
  auto const fail_token = [](headers_t&) -> boost::asio::awaitable<void> {
    throw std::runtime_error{"oauth token request failed"};
    co_return;
  };

  f.fetch(&cache, 3600s, std::nullopt, {}, add_token);
  EXPECT_EQ("Bearer token",
            f.server_.last_request()[http::field::authorization]);

  auto const n_requests = f.server_.srv_.requests_.size();
  auto const r = f.fetch(&cache, 3600s, std::nullopt, {}, fail_token);
  EXPECT_EQ(feed_cache::status::kStale, r.status_);
  EXPECT_EQ("content v1", r.decoded_body_);
  EXPECT_EQ("oauth token request failed", r.download_error_);
  EXPECT_EQ(n_requests, f.server_.srv_.requests_.size());

  EXPECT_THROW(f.fetch(nullptr, 3600s, std::nullopt, {}, fail_token),
               std::runtime_error);
}
