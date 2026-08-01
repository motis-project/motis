#include "motis/rt_update.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#ifdef NO_DATA
#undef NO_DATA
#endif
#include "gtfsrt/gtfs-realtime.pb.h"

#include "utl/read_file.h"
#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/elevators/update_elevators.h"
#include "motis/http_req.h"
#include "motis/railviz.h"
#include "motis/rt/auser.h"
#include "motis/rt/rt_metrics.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
namespace fs = std::filesystem;
using asio::awaitable;

namespace motis {

asio::awaitable<ptr<elevators>> update_elevators(config const& c,
                                                 data const& d,
                                                 n::rt_timetable& new_rtt) {
  utl::verify(c.has_elevators() && c.get_elevators()->url_ && c.timetable_,
              "elevator update requires settings for timetable + elevators");
  auto const res =
      co_await http_GET(boost::urls::url{*c.get_elevators()->url_},
                        c.get_elevators()->headers_.value_or(headers_t{}),
                        std::chrono::seconds{c.get_elevators()->http_timeout_});
  co_return update_elevators(c, d, get_http_body(res), new_rtt);
}

std::string get_dump_path(auto&& ep) {
  auto const normalize = [](std::string const& x) {
    auto ret = std::string{};
    ret.resize(x.size());
    for (auto [to, from] : utl::zip(ret, x)) {
      auto const c = from;
      if (('0' <= c && c <= '9') ||  //
          ('a' <= c && c <= 'z') ||  //
          ('A' <= c && c <= 'Z')) {
        to = c;
      } else {
        to = '_';
      }
    }
    return ret;
  };
  return fmt::format("dump_rt/{}-{}", ep.tag_, normalize(ep.ep_.url_));
}

struct gtfs_rt_endpoint {
  struct last_good {
    transit_realtime::FeedMessage snapshot_;
    bool has_snapshot_{false};
    std::chrono::steady_clock::time_point received_at_{};
    std::chrono::steady_clock::time_point expires_at_{};
    std::chrono::seconds age_at_receipt_{0};
    bool failed_{false};
    bool expired_{false};
  };

  config::timetable::dataset::rt ep_;
  n::source_idx_t src_;
  std::string tag_;
  std::string id_;
  gtfsrt_metrics metrics_;
  std::shared_ptr<last_good> last_good_{std::make_shared<last_good>()};
};

struct auser_endpoint {
  config::timetable::dataset::rt ep_;
  n::source_idx_t src_;
  std::string tag_;
  vdvaus_metrics metrics_;
};

enum struct gtfsrt_payload_error { empty_body, decode_error, missing_header };

class gtfsrt_payload_exception : public std::runtime_error {
public:
  gtfsrt_payload_exception(gtfsrt_payload_error const error,
                           char const* const message)
      : std::runtime_error{message}, error_{error} {}

  gtfsrt_payload_error error_;
};

transit_realtime::FeedMessage validate_gtfsrt_payload(
    std::string_view const body) {
  if (body.empty()) {
    throw gtfsrt_payload_exception{gtfsrt_payload_error::empty_body,
                                   "empty GTFS-RT feed"};
  }
  auto msg = transit_realtime::FeedMessage{};
  if (!msg.ParsePartialFromArray(body.data(), static_cast<int>(body.size()))) {
    throw gtfsrt_payload_exception{gtfsrt_payload_error::decode_error,
                                   "unable to parse GTFS-RT feed"};
  }
  if (!msg.has_header()) {
    throw gtfsrt_payload_exception{gtfsrt_payload_error::missing_header,
                                   "GTFS-RT feed has no header"};
  }
  if (!msg.IsInitialized()) {
    throw gtfsrt_payload_exception{gtfsrt_payload_error::decode_error,
                                   "unable to parse GTFS-RT feed"};
  }
  return msg;
}

void count_payload_error(gtfsrt_metrics const& metrics,
                         gtfsrt_payload_error const error) {
  switch (error) {
    case gtfsrt_payload_error::empty_body:
      metrics.empty_body_.Increment();
      break;
    case gtfsrt_payload_error::decode_error:
      metrics.decode_error_.Increment();
      break;
    case gtfsrt_payload_error::missing_header:
      metrics.missing_header_.Increment();
      break;
  }
}

transit_realtime::FeedMessage materialize_gtfsrt_snapshot(
    transit_realtime::FeedMessage const& base,
    transit_realtime::FeedMessage const& update) {
  auto entities = std::map<std::string, transit_realtime::FeedEntity>{};
  for (auto const& entity : base.entity()) {
    entities.insert_or_assign(entity.id(), entity);
  }
  for (auto const& entity : update.entity()) {
    if (entity.has_is_deleted() && entity.is_deleted()) {
      entities.erase(entity.id());
    } else {
      entities.insert_or_assign(entity.id(), entity);
    }
  }

  auto materialized = update;
  materialized.clear_entity();
  for (auto const& [_, entity] : entities) {
    *materialized.add_entity() = entity;
  }
  materialized.mutable_header()->set_incrementality(
      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
  return materialized;
}

std::chrono::seconds gtfsrt_payload_age(
    transit_realtime::FeedMessage const& msg,
    std::chrono::system_clock::time_point const now) {
  if (!msg.header().has_timestamp() || msg.header().timestamp() == 0U) {
    return std::chrono::seconds{0};
  }
  auto const now_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch())
          .count();
  auto const timestamp = msg.header().timestamp();
  if (now_seconds <= 0 ||
      timestamp >= static_cast<std::uint64_t>(now_seconds)) {
    return std::chrono::seconds{0};
  }
  return std::chrono::seconds{now_seconds -
                              static_cast<std::int64_t>(timestamp)};
}

void run_rt_update(boost::asio::io_context& ioc,
                   config const& c,
                   data& d,
                   rt_update_hooks hooks) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d, hooks = std::move(hooks)]() -> awaitable<void> {
        auto const dump_rt = fs::is_directory("dump_rt");
        if (dump_rt) {
          fmt::println("WARNING: DUMPING TO dump_rt\n");
        }

        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        auto const endpoints = [&]() {
          auto endpoints =
              std::vector<std::variant<gtfs_rt_endpoint, auser_endpoint>>{};
          auto const metric_families =
              rt_metric_families{d.metrics_->registry_};
          for (auto const& [tag, dataset] : c.timetable_->datasets_) {
            if (dataset.rt_.has_value()) {
              auto const src = d.tags_->get_src(tag);
              auto gtfsrt_endpoint_idx = 0U;
              for (auto const& ep : *dataset.rt_) {
                switch (ep.protocol_) {
                  case config::timetable::dataset::rt::protocol::gtfsrt: {
                    auto const endpoint_id =
                        std::to_string(gtfsrt_endpoint_idx++);
                    endpoints.push_back(gtfs_rt_endpoint{
                        ep, src, tag, endpoint_id,
                        gtfsrt_metrics{tag, endpoint_id, metric_families}});
                    break;
                  }
                  case config::timetable::dataset::rt::protocol::siri_json:
                  case config::timetable::dataset::rt::protocol::siri:
                    [[fallthrough]];
                  case config::timetable::dataset::rt::protocol::auser:
                    endpoints.push_back(auser_endpoint{
                        ep, src, tag, vdvaus_metrics{tag, metric_families}});
                    break;
                }
              }
            }
          }
          return endpoints;
        }();
        auto const has_gtfsrt_endpoint =
            std::any_of(endpoints.begin(), endpoints.end(), [](auto const& ep) {
              return std::holds_alternative<gtfs_rt_endpoint>(ep);
            });
        auto const has_auser_endpoint =
            std::any_of(endpoints.begin(), endpoints.end(), [](auto const& ep) {
              return std::holds_alternative<auser_endpoint>(ep);
            });
        auto const rebuild_gtfsrt_from_materialized_snapshots =
            c.timetable_->incremental_rt_update_ && has_gtfsrt_endpoint;
        auto const mixed_incremental_sources =
            rebuild_gtfsrt_from_materialized_snapshots && has_auser_endpoint;
        auto auser_rtt = std::unique_ptr<n::rt_timetable>{};

        while (true) {
          // Remember when we started, so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          try {
            auto t = utl::scoped_timer{"rt update"};

            // Create new real-time timetable.
            auto const today = std::chrono::time_point_cast<date::days>(
                std::chrono::system_clock::now());
            if (mixed_incremental_sources && !auser_rtt) {
              auser_rtt = std::make_unique<n::rt_timetable>(
                  n::rt::create_rt_timetable(*d.tt_, today));
            }
            auto rtt = std::make_unique<n::rt_timetable>(
                c.timetable_->incremental_rt_update_ &&
                        !rebuild_gtfsrt_from_materialized_snapshots
                    ? n::rt_timetable{*d.rt_->rtt_}
                    : n::rt::create_rt_timetable(*d.tt_, today));

            // Schedule updates for each real-time endpoint.
            auto const timeout =
                std::chrono::seconds{c.timetable_->http_timeout_};

            using stats_t =
                std::variant<n::rt::statistics, n::rt::vdv_aus::statistics>;
            struct update_result {
              stats_t stats_;
              bool source_success_{true};
            };
            struct collected_update {
              std::optional<std::string> body_;
              std::exception_ptr error_;
            };
            struct prepared_gtfsrt_update {
              std::size_t endpoint_idx_;
              std::optional<transit_realtime::FeedMessage> msg_;
              bool source_success_{true};
              bool commit_last_good_{false};
            };
            struct prepared_auser_update {
              std::size_t endpoint_idx_;
              std::optional<std::string> body_;
            };
            using prepared_update =
                std::variant<prepared_gtfsrt_update, prepared_auser_update>;
            struct update_group {
              std::string tag_;
              n::source_idx_t src_;
              std::vector<prepared_update> updates_;
            };

            auto const cache_age =
                [](gtfs_rt_endpoint::last_good const& state,
                   std::chrono::steady_clock::time_point const now) {
                  return state.age_at_receipt_ +
                         std::chrono::duration_cast<std::chrono::seconds>(
                             now - state.received_at_);
                };
            auto const expire_cache =
                [&](gtfs_rt_endpoint const& g,
                    std::chrono::steady_clock::time_point const now) {
                  auto& state = *g.last_good_;
                  if (!state.has_snapshot_ || now < state.expires_at_) {
                    return false;
                  }
                  auto const age = cache_age(state, now);
                  if (!state.expired_) {
                    g.metrics_.last_good_expiry_.Increment();
                  }
                  state.has_snapshot_ = false;
                  state.expired_ = true;
                  g.metrics_.set_source_state(gtfsrt_source_state::expired,
                                              static_cast<double>(age.count()),
                                              false);
                  return true;
                };
            auto const commit_last_good =
                [&](gtfs_rt_endpoint const& g,
                    transit_realtime::FeedMessage candidate) {
                  auto& state = *g.last_good_;
                  auto const received_at = std::chrono::steady_clock::now();
                  auto const age_at_receipt = gtfsrt_payload_age(
                      candidate, std::chrono::system_clock::now());
                  auto const ttl = std::chrono::seconds{g.ep_.last_good_ttl_};
                  if (state.failed_ || state.expired_) {
                    g.metrics_.recovery_.Increment();
                  }
                  state.snapshot_ = std::move(candidate);
                  state.has_snapshot_ = true;
                  state.received_at_ = received_at;
                  state.age_at_receipt_ = age_at_receipt;
                  state.expires_at_ = received_at + ttl - age_at_receipt;
                  state.failed_ = false;
                  state.expired_ = false;
                  g.metrics_.set_source_state(
                      gtfsrt_source_state::live,
                      static_cast<double>(age_at_receipt.count()), true);
                };
            auto const prepare_last_good = [&](std::size_t const endpoint_idx,
                                               gtfs_rt_endpoint const& g) {
              auto const now = std::chrono::steady_clock::now();
              expire_cache(g, now);
              if (g.last_good_->has_snapshot_) {
                g.last_good_->failed_ = true;
                g.metrics_.last_good_reuse_.Increment();
                g.metrics_.set_source_state(
                    gtfsrt_source_state::replay,
                    static_cast<double>(cache_age(*g.last_good_, now).count()),
                    true);
                return prepared_gtfsrt_update{
                    endpoint_idx, g.last_good_->snapshot_, false, false};
              }
              g.last_good_->failed_ = true;
              auto const expired = g.last_good_->expired_;
              auto const age = expired
                                   ? static_cast<double>(
                                         cache_age(*g.last_good_, now).count())
                                   : 0.0;
              g.metrics_.set_source_state(expired
                                              ? gtfsrt_source_state::expired
                                              : gtfsrt_source_state::no_base,
                                          age, false);
              return prepared_gtfsrt_update{endpoint_idx, std::nullopt, false,
                                            false};
            };
            auto const prepare_valid_gtfsrt =
                [&](std::size_t const endpoint_idx, gtfs_rt_endpoint const& g,
                    std::string_view const body) {
                  auto msg = validate_gtfsrt_payload(body);
                  auto const differential =
                      msg.header().incrementality() ==
                      transit_realtime::FeedHeader_Incrementality_DIFFERENTIAL;
                  expire_cache(g, std::chrono::steady_clock::now());
                  auto candidate = differential
                                       ? materialize_gtfsrt_snapshot(
                                             g.last_good_->snapshot_, msg)
                                       : std::move(msg);
                  candidate.mutable_header()->set_incrementality(
                      transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
                  auto const age_at_receipt = gtfsrt_payload_age(
                      candidate, std::chrono::system_clock::now());
                  if (age_at_receipt >=
                      std::chrono::seconds{g.ep_.last_good_ttl_}) {
                    if (!c.timetable_->canned_rt_) {
                      g.metrics_.updates_error_.Increment();
                    }
                    auto& state = *g.last_good_;
                    state.failed_ = true;
                    if (state.has_snapshot_) {
                      return prepare_last_good(endpoint_idx, g);
                    }
                    if (state.expired_) {
                      g.metrics_.set_source_state(
                          gtfsrt_source_state::expired,
                          static_cast<double>(cache_age(
                                                  state,
                                                  std::chrono::steady_clock::now())
                                                  .count()),
                          false);
                      return prepared_gtfsrt_update{
                          endpoint_idx, std::nullopt, false, false};
                    }
                    g.metrics_.last_good_expiry_.Increment();
                    state.snapshot_.Clear();
                    state.has_snapshot_ = false;
                    state.received_at_ = std::chrono::steady_clock::now();
                    state.expires_at_ = state.received_at_;
                    state.age_at_receipt_ = age_at_receipt;
                    state.expired_ = true;
                    g.metrics_.set_source_state(
                        gtfsrt_source_state::expired,
                        static_cast<double>(age_at_receipt.count()), false);
                    return prepared_gtfsrt_update{
                        endpoint_idx, std::nullopt, false, false};
                  }
                  return prepared_gtfsrt_update{
                      endpoint_idx, std::move(candidate), true, true};
                };
            auto const apply_gtfsrt =
                [&](gtfs_rt_endpoint const& g,
                    prepared_gtfsrt_update const& prepared,
                    transit_realtime::FeedMessage const& msg,
                    bool const fallback) {
                  // GTFS-RT application mutates incrementally and can throw.
                  // Apply to a private copy so a failed primary or fallback
                  // cannot leak a partially changed timetable into this cycle.
                  auto staged = *rtt;
                  auto stats = n::rt::gtfsrt_update_msg(*d.tt_, staged, g.src_,
                                                        g.tag_, msg);
                  if (hooks.after_gtfsrt_apply_) {
                    hooks.after_gtfsrt_apply_(prepared.endpoint_idx_, fallback);
                  }
                  if (prepared.commit_last_good_) {
                    commit_last_good(g, msg);
                  }
                  rtt = std::make_unique<n::rt_timetable>(std::move(staged));
                  return stats;
                };

            // Collect every response before parsing or mutating the timetable.
            auto collected = std::vector<collected_update>{};
            collected.reserve(endpoints.size());
            if (c.timetable_->canned_rt_) {
              fmt::println("WARNING: READING CANNED RT");
              for (auto const& ep : endpoints) {
                auto const path = std::visit(
                    [](auto const& x) { return get_dump_path(x); }, ep);
                collected.push_back({.body_ = utl::read_file(path.c_str())});
              }
            } else if (!endpoints.empty()) {
              auto awaitables = utl::to_vec(
                  endpoints,
                  [&](std::variant<gtfs_rt_endpoint, auser_endpoint> const& x) {
                    return boost::asio::co_spawn(
                        executor,
                        [&, endpoint = &x]() -> awaitable<std::string> {
                          co_return co_await std::visit(
                              utl::overloaded{
                                  [&](gtfs_rt_endpoint const& g)
                                      -> awaitable<std::string> {
                                    g.metrics_.updates_requested_.Increment();
                                    auto const res = co_await http_GET(
                                        boost::urls::url{g.ep_.url_},
                                        g.ep_.headers_.value_or(headers_t{}),
                                        timeout);
                                    auto body = get_http_body(res);
                                    if (dump_rt) {
                                      std::ofstream{get_dump_path(g),
                                                    std::ios::binary}
                                          .write(body.data(), static_cast<long>(
                                                                  body.size()));
                                    }
                                    co_return body;
                                  },
                                  [&](auser_endpoint const& a)
                                      -> awaitable<std::string> {
                                    a.metrics_.updates_requested_.Increment();
                                    auto& auser = d.auser_->at(a.ep_.url_);
                                    auto const fetch_url = boost::urls::url{
                                        auser.fetch_url(a.ep_.url_)};
                                    fmt::println("[auser] fetch url: {}",
                                                 fetch_url.c_str());
                                    auto const res = co_await http_GET(
                                        fetch_url,
                                        a.ep_.headers_.value_or(headers_t{}),
                                        timeout);
                                    auto body = get_http_body(res);
                                    if (dump_rt) {
                                      std::ofstream{get_dump_path(a),
                                                    std::ios::binary}
                                          .write(body.data(), static_cast<long>(
                                                                  body.size()));
                                    }
                                    co_return body;
                                  }},
                              *endpoint);
                        },
                        asio::deferred);
                  });

              auto [_, exceptions, bodies] =
                  co_await asio::experimental::make_parallel_group(awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);
              for (auto&& [ex, body] : utl::zip(exceptions, bodies)) {
                if (ex) {
                  collected.push_back({std::nullopt, ex});
                } else {
                  collected.push_back({std::move(body), {}});
                }
              }
            }

            auto results = std::vector<update_result>(endpoints.size());
            auto groups = std::vector<update_group>{};
            auto const add_to_group = [&](std::size_t const endpoint_idx,
                                          prepared_update update) {
              auto const [tag, src] = std::visit(
                  [](auto const& ep) { return std::pair{ep.tag_, ep.src_}; },
                  endpoints[endpoint_idx]);
              auto const it = std::find_if(
                  groups.begin(), groups.end(), [&](update_group const& group) {
                    return group.tag_ == tag && group.src_ == src;
                  });
              auto& group =
                  it == groups.end() ? groups.emplace_back(tag, src) : *it;
              group.updates_.push_back(std::move(update));
            };

            // Parse and group in configured endpoint order. Network completion
            // order is deliberately absent from this phase.
            for (auto endpoint_idx = std::size_t{0};
                 endpoint_idx != endpoints.size(); ++endpoint_idx) {
              auto const& ep = endpoints[endpoint_idx];
              auto& fetched = collected[endpoint_idx];
              utl::visit(
                  ep,
                  [&](gtfs_rt_endpoint const& g) {
                    try {
                      if (fetched.error_) {
                        std::rethrow_exception(fetched.error_);
                      }
                      if (!fetched.body_) {
                        throw std::runtime_error{
                            "GTFS-RT fetch returned no "
                            "payload"};
                      }
                      add_to_group(endpoint_idx,
                                   prepare_valid_gtfsrt(endpoint_idx, g,
                                                        *fetched.body_));
                    } catch (gtfsrt_payload_exception const& e) {
                      count_payload_error(g.metrics_, e.error_);
                      if (!c.timetable_->canned_rt_) {
                        g.metrics_.updates_error_.Increment();
                      }
                      n::log(n::log_lvl::error, "motis.rt",
                             "RT PAYLOAD ERROR: tag={}, error={}", g.tag_,
                             e.what());
                      add_to_group(endpoint_idx,
                                   prepare_last_good(endpoint_idx, g));
                    } catch (std::exception const& e) {
                      g.metrics_.fetch_error_.Increment();
                      if (!c.timetable_->canned_rt_) {
                        g.metrics_.updates_error_.Increment();
                      }
                      n::log(n::log_lvl::error, "motis.rt",
                             "RT FETCH ERROR: tag={}, error={}", g.tag_,
                             e.what());
                      add_to_group(endpoint_idx,
                                   prepare_last_good(endpoint_idx, g));
                    }
                  },
                  [&](auser_endpoint const& a) {
                    if (fetched.error_ || !fetched.body_) {
                      if (fetched.error_) {
                        try {
                          std::rethrow_exception(fetched.error_);
                        } catch (std::exception const& e) {
                          n::log(n::log_lvl::error, "motis.rt",
                                 "VDV AUS FETCH ERROR: tag={}, url={}, "
                                 "error={}",
                                 a.tag_, a.ep_.url_, e.what());
                        }
                      }
                      if (!c.timetable_->canned_rt_) {
                        a.metrics_.updates_error_.Increment();
                      }
                      results[endpoint_idx] = {
                          n::rt::vdv_aus::statistics{.error_ = true}, false};
                    }
                    add_to_group(endpoint_idx,
                                 prepared_auser_update{
                                     endpoint_idx, std::move(fetched.body_)});
                  });
            }

            auto const apply_prepared_gtfsrt =
                [&](prepared_gtfsrt_update& prepared) {
                      auto const& g = std::get<gtfs_rt_endpoint>(
                          endpoints[prepared.endpoint_idx_]);
                      if (!prepared.msg_) {
                        results[prepared.endpoint_idx_] = {
                            n::rt::statistics{.parser_error_ = true}, false};
                        return;
                      }
                      try {
                        auto stats =
                            apply_gtfsrt(g, prepared, *prepared.msg_, false);
                        results[prepared.endpoint_idx_] = {
                            std::move(stats), prepared.source_success_};
                      } catch (std::exception const& e) {
                        if (!c.timetable_->canned_rt_) {
                          g.metrics_.updates_error_.Increment();
                        }
                        n::log(n::log_lvl::error, "motis.rt",
                               "RT APPLY ERROR: tag={}, error={}", g.tag_,
                               e.what());
                        auto fallback =
                            prepare_last_good(prepared.endpoint_idx_, g);
                        if (fallback.msg_) {
                          try {
                            results[prepared.endpoint_idx_] = {
                                apply_gtfsrt(g, fallback, *fallback.msg_, true),
                                false};
                          } catch (std::exception const& fallback_error) {
                            if (!c.timetable_->canned_rt_) {
                              g.metrics_.updates_error_.Increment();
                            }
                            n::log(n::log_lvl::error, "motis.rt",
                                   "RT FALLBACK APPLY ERROR: tag={}, error={}",
                                   g.tag_, fallback_error.what());
                            results[prepared.endpoint_idx_] = {
                                n::rt::statistics{.parser_error_ = true},
                                false};
                          } catch (...) {
                            if (!c.timetable_->canned_rt_) {
                              g.metrics_.updates_error_.Increment();
                            }
                            n::log(n::log_lvl::error, "motis.rt",
                                   "RT FALLBACK APPLY ERROR: tag={}, "
                                   "error=unknown",
                                   g.tag_);
                            results[prepared.endpoint_idx_] = {
                                n::rt::statistics{.parser_error_ = true},
                                false};
                          }
                        } else {
                          results[prepared.endpoint_idx_] = {
                              n::rt::statistics{.parser_error_ = true}, false};
                        }
                      }
                };
            auto const apply_prepared_auser =
                [&](prepared_auser_update& prepared) {
                      if (!prepared.body_) {
                        return;
                      }
                      auto const& a = std::get<auser_endpoint>(
                          endpoints[prepared.endpoint_idx_]);
                      try {
                        auto& auser = d.auser_->at(a.ep_.url_);
                        auto& target = mixed_incremental_sources
                                           ? *auser_rtt
                                           : *rtt;
                        results[prepared.endpoint_idx_] = {
                            auser.consume_update(*prepared.body_, target, true)};
                      } catch (std::exception const& e) {
                        if (!c.timetable_->canned_rt_) {
                          a.metrics_.updates_error_.Increment();
                        }
                        n::log(n::log_lvl::error, "motis.rt",
                               "VDV AUS APPLY ERROR: tag={}, url={}, error={}",
                               a.tag_, a.ep_.url_, e.what());
                        results[prepared.endpoint_idx_] = {
                            n::rt::vdv_aus::statistics{.error_ = true}, false};
                      }
                };

            // Apply every prepared provider message once, serially and in
            // configured order within its dataset/source group. Mixed
            // incremental deployments first advance their persistent
            // AUSER/SIRI baseline, then overlay GTFS snapshots on a copy.
            if (mixed_incremental_sources) {
              for (auto& group : groups) {
                for (auto& update : group.updates_) {
                  if (auto* prepared =
                          std::get_if<prepared_auser_update>(&update);
                      prepared != nullptr) {
                    apply_prepared_auser(*prepared);
                  }
                }
              }
              rtt = std::make_unique<n::rt_timetable>(*auser_rtt);
              for (auto& group : groups) {
                for (auto& update : group.updates_) {
                  if (auto* prepared =
                          std::get_if<prepared_gtfsrt_update>(&update);
                      prepared != nullptr) {
                    apply_prepared_gtfsrt(*prepared);
                  }
                }
              }
            } else {
              for (auto& group : groups) {
                for (auto& update : group.updates_) {
                  utl::visit(update, apply_prepared_gtfsrt,
                             apply_prepared_auser);
                }
              }
            }

            for (auto&& [ep, result] : utl::zip(endpoints, results)) {
              utl::visit(
                  ep,
                  [&](gtfs_rt_endpoint const& g) {
                    auto const& stats =
                        std::get<n::rt::statistics>(result.stats_);
                    if (!c.timetable_->canned_rt_ && result.source_success_) {
                      g.metrics_.updates_successful_.Increment();
                      g.metrics_.last_update_timestamp_.SetToCurrentTime();
                      g.metrics_.update(stats);
                    }
                    n::log(n::log_lvl::info, "motis.rt",
                           "GTFS-RT update stats for tag={}, url={}: {}",
                           g.tag_, g.ep_.url_, fmt::streamed(stats));
                  },
                  [&](auser_endpoint const& a) {
                    auto const& stats =
                        std::get<n::rt::vdv_aus::statistics>(result.stats_);
                    if (!c.timetable_->canned_rt_ && result.source_success_) {
                      a.metrics_.updates_successful_.Increment();
                      a.metrics_.last_update_timestamp_.SetToCurrentTime();
                      a.metrics_.update(stats);
                    }
                    n::log(n::log_lvl::info, "motis.rt",
                           "VDV AUS update stats for tag={}, url={}:\n{}",
                           a.tag_, a.ep_.url_, fmt::streamed(stats));
                  });
            }

            // Update lbs.
            rtt->update_lbs(*d.tt_);

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators = c.has_elevators() && c.get_elevators()->url_
                                 ? co_await update_elevators(c, d, *rtt)
                                 : std::move(d.rt_->e_);
            auto new_rt = std::make_shared<rt>(
                std::move(rtt), std::move(elevators), std::move(railviz_rt));
            std::atomic_store(&d.rt_, std::move(new_rt));

            d.metrics_->last_update_rt_.SetToCurrentTime();
          } catch (std::exception const& e) {
            n::log(n::log_lvl::error, "motis.rt",
                   "RT UPDATE CYCLE ERROR: error={}", e.what());
          } catch (...) {
            n::log(n::log_lvl::error, "motis.rt",
                   "RT UPDATE CYCLE ERROR: error=unknown");
          }

          // Schedule next update.
          timer.expires_at(
              start + std::chrono::seconds{c.timetable_->update_interval_});
          co_await timer.async_wait(
              asio::redirect_error(asio::use_awaitable, ec));
          if (ec == asio::error::operation_aborted) {
            co_return;
          }
        }
      },
      boost::asio::detached);
}

}  // namespace motis
