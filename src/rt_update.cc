#include "motis/rt_update.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
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

struct gtfsrt_payload_exception final : std::runtime_error {
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
                        ep, src, tag,
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
        auto auser_rtt_day = std::optional<date::sys_days>{};

        while (true) {
          // Remember when we started, so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          {
            auto t = utl::scoped_timer{"rt update"};

            // Create new real-time timetable.
            auto const now = hooks.now_ ? hooks.now_()
                                        : std::chrono::system_clock::now();
            auto const today = std::chrono::time_point_cast<date::days>(now);
            auto const auser_day_rollover =
                has_auser_endpoint &&
                (mixed_incremental_sources
                     ? auser_rtt_day != today
                     : d.rt_->rtt_->base_day_ != today);
            if (auser_day_rollover) {
              auto reset_urls = std::set<std::string_view>{};
              for (auto const& endpoint : endpoints) {
                if (auto const* a = std::get_if<auser_endpoint>(&endpoint);
                    a != nullptr && reset_urls.emplace(a->ep_.url_).second) {
                  d.auser_->at(a->ep_.url_).reset_for_resync();
                }
              }
            }
            if (mixed_incremental_sources && auser_rtt_day != today) {
              auser_rtt = std::make_unique<n::rt_timetable>(
                  n::rt::create_rt_timetable(*d.tt_, today));
              auser_rtt_day = today;
            }
            auto rtt = std::make_unique<n::rt_timetable>(
                c.timetable_->incremental_rt_update_ &&
                        !rebuild_gtfsrt_from_materialized_snapshots &&
                        !auser_day_rollover
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

            auto const apply_gtfsrt_update =
                [&](gtfs_rt_endpoint const& g,
                    std::string_view const body) -> update_result {
              return {
                  n::rt::gtfsrt_update_buf(*d.tt_, *rtt, g.src_, g.tag_, body)};
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
                  g.metrics_.last_good_expiry_.Increment();
                  state.has_snapshot_ = false;
                  state.expired_ = true;
                  g.metrics_.set_source_state(gtfsrt_source_state::expired,
                                              static_cast<double>(age.count()),
                                              false);
                  return true;
                };
            auto const commit_last_good =
                [&](gtfs_rt_endpoint const& g,
                    transit_realtime::FeedMessage candidate,
                    std::chrono::seconds const age_at_receipt) {
                  auto& state = *g.last_good_;
                  auto const received_at = std::chrono::steady_clock::now();
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
            auto const reuse_last_good =
                [&](gtfs_rt_endpoint const& g) -> update_result {
              auto const now = std::chrono::steady_clock::now();
              expire_cache(g, now);
              if (g.last_good_->has_snapshot_) {
                g.last_good_->failed_ = true;
                g.metrics_.last_good_reuse_.Increment();
                auto const payload =
                    g.last_good_->snapshot_.SerializeAsString();
                auto result = apply_gtfsrt_update(g, payload);
                g.metrics_.set_source_state(
                    gtfsrt_source_state::replay,
                    static_cast<double>(cache_age(*g.last_good_, now).count()),
                    true);
                result.source_success_ = false;
                return result;
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
              return {n::rt::statistics{.parser_error_ = true}, false};
            };
            auto const reject_stale_candidate =
                [&](gtfs_rt_endpoint const& g,
                    std::chrono::seconds const age_at_receipt) {
                  auto& state = *g.last_good_;
                  state.failed_ = true;
                  if (state.has_snapshot_) {
                    return reuse_last_good(g);
                  }
                  if (state.expired_) {
                    g.metrics_.set_source_state(
                        gtfsrt_source_state::expired,
                        static_cast<double>(cache_age(
                                                state,
                                                std::chrono::steady_clock::now())
                                                .count()),
                        false);
                    return update_result{
                        n::rt::statistics{.parser_error_ = true}, false};
                  }
                  if (!state.expired_) {
                    g.metrics_.last_good_expiry_.Increment();
                  }
                  state.snapshot_.Clear();
                  state.has_snapshot_ = false;
                  state.received_at_ = std::chrono::steady_clock::now();
                  state.expires_at_ = state.received_at_;
                  state.age_at_receipt_ = age_at_receipt;
                  state.expired_ = true;
                  g.metrics_.set_source_state(
                      gtfsrt_source_state::expired,
                      static_cast<double>(age_at_receipt.count()), false);
                  return update_result{n::rt::statistics{.parser_error_ = true},
                                       false};
                };
            auto const apply_valid_update =
                [&](gtfs_rt_endpoint const& g,
                    std::string_view const body) -> update_result {
              auto validation = validate_gtfsrt_payload(body);
              auto const differential =
                  validation.header().incrementality() ==
                  transit_realtime::FeedHeader_Incrementality_DIFFERENTIAL;
              expire_cache(g, std::chrono::steady_clock::now());
              auto candidate = differential
                                   ? materialize_gtfsrt_snapshot(
                                         g.last_good_->snapshot_, validation)
                                   : std::move(validation);
              candidate.mutable_header()->set_incrementality(
                  transit_realtime::FeedHeader_Incrementality_FULL_DATASET);
              auto const age_at_receipt = gtfsrt_payload_age(
                  candidate, std::chrono::system_clock::now());
              if (age_at_receipt >=
                  std::chrono::seconds{g.ep_.last_good_ttl_}) {
                g.metrics_.updates_error_.Increment();
                return reject_stale_candidate(g, age_at_receipt);
              }
              auto const materialized = candidate.SerializeAsString();

              auto result = apply_gtfsrt_update(g, materialized);
              commit_last_good(g, std::move(candidate), age_at_receipt);
              return result;
            };
            if (c.timetable_->canned_rt_) {
              fmt::println("WARNING: READING CANNED RT");

              auto const stats =
                  utl::to_vec(endpoints, [&](auto&& ep) -> update_result {
                    try {
                      return utl::visit(
                          ep,
                          [&](gtfs_rt_endpoint const& g) -> update_result {
                            auto const path = get_dump_path(g);
                            auto const body = utl::read_file(path.c_str());
                            if (body.has_value()) {
                              return apply_valid_update(g, *body);
                            } else {
                              g.metrics_.fetch_error_.Increment();
                              return reuse_last_good(g);
                            }
                          },
                          [&](auser_endpoint const& a) -> update_result {
                            auto const path = get_dump_path(a);
                            auto& auser = d.auser_->at(a.ep_.url_);
                            auto const body = utl::read_file(path.c_str());
                            if (body.has_value()) {
                              auto& target = mixed_incremental_sources
                                                 ? *auser_rtt
                                                 : *rtt;
                              return {auser.consume_update(*body, target)};
                            } else {
                              return {
                                  n::rt::vdv_aus::statistics{.error_ = true}};
                            }
                          });
                    } catch (gtfsrt_payload_exception const& e) {
                      std::cout << "EXCEPTION: " << e.what() << "\n";
                      return utl::visit(
                          ep,
                          [&](gtfs_rt_endpoint const& g) {
                            count_payload_error(g.metrics_, e.error_);
                            return reuse_last_good(g);
                          },
                          [&](auser_endpoint const&) {
                            return update_result{
                                n::rt::statistics{.parser_error_ = true},
                                false};
                          });
                    } catch (std::exception const& e) {
                      std::cout << "EXCEPTION: " << e.what() << "\n";
                      return utl::visit(
                          ep,
                          [&](gtfs_rt_endpoint const& g) {
                            return reuse_last_good(g);
                          },
                          [&](auser_endpoint const&) {
                            return update_result{
                                n::rt::statistics{.parser_error_ = true},
                                false};
                          });
                    }
                  });

              for (auto const [s, ep] : utl::zip(stats, endpoints)) {
                utl::visit(
                    ep,
                    [&](gtfs_rt_endpoint const& g) {
                      n::log(
                          n::log_lvl::info, "motis.rt",
                          "GTFS-RT update stats for tag={}, url={}: {}", g.tag_,
                          g.ep_.url_,
                          fmt::streamed(std::get<n::rt::statistics>(s.stats_)));
                    },
                    [&](auser_endpoint const& a) {
                      n::log(n::log_lvl::info, "motis.rt",
                             "VDV AUS update stats for tag={}, url={}:\n{}",
                             a.tag_, a.ep_.url_,
                             fmt::streamed(std::get<n::rt::vdv_aus::statistics>(
                                 s.stats_)));
                    });
              }
            } else if (!endpoints.empty()) {
              auto awaitables = utl::to_vec(
                  endpoints,
                  [&](std::variant<gtfs_rt_endpoint, auser_endpoint> const& x) {
                    return boost::asio::co_spawn(
                        executor,
                        [&]() -> awaitable<update_result> {
                          auto ret = update_result{};
                          co_await std::visit(
                              utl::overloaded{
                                  [&](gtfs_rt_endpoint const& g)
                                      -> awaitable<void> {
                                    g.metrics_.updates_requested_.Increment();
                                    try {
                                      auto const res = co_await http_GET(
                                          boost::urls::url{g.ep_.url_},
                                          g.ep_.headers_.value_or(headers_t{}),
                                          timeout);
                                      auto const body = get_http_body(res);
                                      if (dump_rt) {
                                        std::ofstream{get_dump_path(g)}.write(
                                            body.c_str(),
                                            static_cast<long>(body.size()));
                                      }
                                      try {
                                        ret = apply_valid_update(g, body);
                                      } catch (
                                          gtfsrt_payload_exception const& e) {
                                        count_payload_error(g.metrics_,
                                                            e.error_);
                                        g.metrics_.updates_error_.Increment();
                                        n::log(n::log_lvl::error, "motis.rt",
                                               "RT PAYLOAD ERROR: tag={}, "
                                               "error={}",
                                               g.tag_, e.what());
                                        ret = reuse_last_good(g);
                                      } catch (std::exception const& e) {
                                        g.metrics_.updates_error_.Increment();
                                        n::log(
                                            n::log_lvl::error, "motis.rt",
                                            "RT APPLY ERROR: tag={}, error={}",
                                            g.tag_, e.what());
                                        ret = reuse_last_good(g);
                                      }
                                    } catch (std::exception const& e) {
                                      g.metrics_.updates_error_.Increment();
                                      g.metrics_.fetch_error_.Increment();
                                      n::log(n::log_lvl::error, "motis.rt",
                                             "RT FETCH ERROR: tag={}, error={}",
                                             g.tag_, e.what());
                                      ret = reuse_last_good(g);
                                    }
                                  },
                                  [&](auser_endpoint const& a)
                                      -> awaitable<void> {
                                    a.metrics_.updates_requested_.Increment();
                                    auto& auser = d.auser_->at(a.ep_.url_);
                                    try {
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
                                        std::ofstream{get_dump_path(a)}.write(
                                            body.c_str(),
                                            static_cast<long>(body.size()));
                                      }
                                      auto& target = mixed_incremental_sources
                                                         ? *auser_rtt
                                                         : *rtt;
                                      ret = {auser.consume_update(body, target,
                                                                  true)};
                                    } catch (std::exception const& e) {
                                      a.metrics_.updates_error_.Increment();
                                      n::log(n::log_lvl::error, "motis.rt",
                                             "VDV AUS FETCH ERROR: tag={}, "
                                             "url={}, error={}",
                                             a.tag_, a.ep_.url_, e.what());
                                      ret = {nigiri::rt::vdv_aus::statistics{
                                          .error_ = true}};
                                    }
                                  }},
                              x);
                          co_return ret;
                        },
                        asio::deferred);
                  });

              // Wait for all updates to finish
              auto [_, exceptions, stats] =
                  co_await asio::experimental::make_parallel_group(awaitables)
                      .async_wait(asio::experimental::wait_for_all(),
                                  asio::use_awaitable);

              //  Print statistics.
              for (auto const [ep, ex, s] :
                   utl::zip(endpoints, exceptions, stats)) {
                std::visit(
                    utl::overloaded{
                        [&](gtfs_rt_endpoint const& g) {
                          try {
                            if (ex) {
                              std::rethrow_exception(ex);
                            }

                            if (s.source_success_) {
                              g.metrics_.updates_successful_.Increment();
                              g.metrics_.last_update_timestamp_
                                  .SetToCurrentTime();
                              g.metrics_.update(
                                  std::get<n::rt::statistics>(s.stats_));
                            }

                            n::log(
                                n::log_lvl::info, "motis.rt",
                                "GTFS-RT update stats for tag={}, url={}: {}",
                                g.tag_, g.ep_.url_,
                                fmt::streamed(
                                    std::get<n::rt::statistics>(s.stats_)));
                          } catch (std::exception const& e) {
                            g.metrics_.updates_error_.Increment();
                            n::log(n::log_lvl::error, "motis.rt",
                                   "GTFS-RT update failed: tag={}, url={}, "
                                   "error={}",
                                   g.tag_, g.ep_.url_, e.what());
                          }
                        },
                        [&](auser_endpoint const& a) {
                          try {
                            if (ex) {
                              std::rethrow_exception(ex);
                            }

                            a.metrics_.updates_successful_.Increment();
                            a.metrics_.last_update_timestamp_
                                .SetToCurrentTime();
                            a.metrics_.update(
                                std::get<n::rt::vdv_aus::statistics>(s.stats_));

                            n::log(
                                n::log_lvl::info, "motis.rt",
                                "VDV AUS update stats for tag={}, url={}:\n{}",
                                a.tag_, a.ep_.url_,
                                fmt::streamed(
                                    std::get<n::rt::vdv_aus::statistics>(
                                        s.stats_)));
                          } catch (std::exception const& e) {
                            a.metrics_.updates_error_.Increment();
                            n::log(n::log_lvl::error, "motis.rt",
                                   "VDV AUS update failed: tag={}, url={}, "
                                   "error={}",
                                   a.tag_, a.ep_.url_, e.what());
                          }
                        }},
                    ep);
              }
            }

            if (mixed_incremental_sources) {
              rtt = std::make_unique<n::rt_timetable>(*auser_rtt);
              for (auto const& ep : endpoints) {
                if (auto const* g = std::get_if<gtfs_rt_endpoint>(&ep);
                    g != nullptr) {
                  expire_cache(*g, std::chrono::steady_clock::now());
                  if (g->last_good_->has_snapshot_) {
                    apply_gtfsrt_update(
                        *g, g->last_good_->snapshot_.SerializeAsString());
                  }
                }
              }
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
