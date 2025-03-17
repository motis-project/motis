#include "motis/rt_update.h"

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"
#include "boost/beast/core/buffers_to_string.hpp"

#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/constants.h"
#include "motis/data.h"
#include "motis/elevators/elevators.h"
#include "motis/elevators/parse_fasta.h"
#include "motis/http_req.h"
#include "motis/railviz.h"
#include "motis/tag_lookup.h"
#include "motis/update_rtt_td_footpaths.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

using elevator_map_t = hash_map<std::int64_t, elevator_idx_t>;

elevator_map_t to_map(vector_map<elevator_idx_t, elevator> const& elevators) {
  auto m = elevator_map_t{};
  for (auto const [i, e] : utl::enumerate(elevators)) {
    m.emplace(e.id_, elevator_idx_t{i});
  }
  return m;
}

asio::awaitable<ptr<elevators>> update_elevators(
    config::elevators const& config, data const& d, n::rt_timetable& new_rtt) {
  auto const res = co_await http_GET(
      boost::urls::url{config.url_}, config.headers_.value_or(headers_t{}),
      std::chrono::seconds{config.http_timeout_});
  auto const body = get_http_body(res);
  auto new_e = std::make_unique<motis::elevators>(
      *d.w_, *d.elevator_nodes_, parse_fasta(std::string_view{body}));

  auto const& old_e = *d.rt_->e_;
  auto const old_map = to_map(old_e.elevators_);
  auto const new_map = to_map(new_e->elevators_);

  auto tasks = hash_set<std::pair<n::location_idx_t, osr::direction>>{};
  auto const add_tasks = [&](geo::latlng const& l) {
    d.location_rtee_->in_radius(l, kElevatorUpdateRadius,
                                [&](n::location_idx_t const l) {
                                  tasks.emplace(l, osr::direction::kForward);
                                  tasks.emplace(l, osr::direction::kBackward);
                                });
  };

  for (auto const [id, e_idx] : old_map) {
    auto const it = new_map.find(id);
    if (it == end(new_map)) {
      // Elevator got removed.
      // Not listed in new => default status = ACTIVE
      // Update if INACTIVE before (= status changed)
      if (old_e.elevators_[e_idx].status_ == false) {
        add_tasks(old_e.elevators_[e_idx].pos_);
      }
    } else {
      // Elevator remained. Update if status changed.
      if (new_e->elevators_[it->second].status_ !=
          old_e.elevators_[e_idx].status_) {
        add_tasks(new_e->elevators_[it->second].pos_);
      }
    }
  }

  for (auto const [id, e_idx] : new_map) {
    auto const it = old_map.find(id);
    if (it == end(old_map)) {
      // New elevator not seen before. Update.
      add_tasks(new_e->elevators_[e_idx].pos_);
    }
  }

  n::log(n::log_lvl::info, "motis.rt.elevators",
         "elevator update: {} routing tasks", tasks.size());

  update_rtt_td_footpaths(*d.w_, *d.l_, *d.pl_, *d.tt_, *d.location_rtee_,
                          *new_e, *d.matches_, tasks, d.rt_->rtt_.get(),
                          new_rtt, std::chrono::seconds{kMaxDuration});

  co_return std::move(new_e);
}

void run_rt_update(boost::asio::io_context& ioc, config const& c, data& d) {
  boost::asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto msg = transit_realtime::FeedMessage{};
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};
        while (true) {
          // Remember when we started, so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          {
            auto t = utl::scoped_timer{"rt update"};

            // Create new real-time timetable.
            auto const today = std::chrono::time_point_cast<date::days>(
                std::chrono::system_clock::now());
            auto rtt = std::make_unique<n::rt_timetable>(
                c.timetable_->incremental_rt_update_
                    ? n::rt_timetable{*d.rt_->rtt_}
                    : n::rt::create_rt_timetable(*d.tt_, today));

            // Schedule updates for each real-time endpoint.
            auto const timeout =
                std::chrono::seconds{c.timetable_->http_timeout_};
            auto endpoints =
                std::vector<std::tuple<config::timetable::dataset::rt,
                                       n::source_idx_t, std::string>>{};
            for (auto const& [tag, dataset] : c.timetable_->datasets_) {
              if (dataset.rt_.has_value()) {
                auto const src = d.tags_->get_src(tag);
                for (auto const& ep : *dataset.rt_) {
                  endpoints.emplace_back(ep, src, tag);
                }
              }
            }

            auto awaitables = utl::to_vec(endpoints, [&](auto&& x) {
              auto const& [ep, src, tag] = x;
              return boost::asio::co_spawn(
                  executor,
                  [ep, src, tag, timeout, &rtt, &msg,
                   &d]() -> awaitable<n::rt::statistics> {
                    try {
                      auto const res = co_await http_GET(
                          boost::urls::url{ep.url_},
                          ep.headers_.value_or(headers_t{}), timeout);
                      co_return n::rt::gtfsrt_update_buf(
                          *d.tt_, *rtt, src, tag, get_http_body(res), msg);
                    } catch (std::exception const& e) {
                      n::log(n::log_lvl::error, "motis.rt",
                             "RT FETCH ERROR: tag={}, error={}", tag, e.what());
                      co_return n::rt::statistics{.parser_error_ = true,
                                                  .no_header_ = true};
                    }
                  },
                  asio::deferred);
            });

            // Wait for all updates to finish
            auto [idx, exceptions, stats] =
                co_await asio::experimental::make_parallel_group(awaitables)
                    .async_wait(asio::experimental::wait_for_all(),
                                asio::use_awaitable);

            //  Print statistics.
            for (auto const [i, ex, s] : utl::zip(idx, exceptions, stats)) {
              auto const [ep, src, tag] = endpoints[i];
              try {
                if (ex) {
                  std::rethrow_exception(ex);
                }
                n::log(n::log_lvl::info, "motis.rt",
                       "rt update stats for tag={}, url={}: {}", tag, ep.url_,
                       fmt::streamed(s));
              } catch (std::exception const& e) {
                n::log(n::log_lvl::error, "motis.rt",
                       "rt update failed: tag={}, url={}, error={}", tag,
                       ep.url_, e.what());
              }
            }

            // Update real-time timetable shared pointer.
            auto railviz_rt = std::make_unique<railviz_rt_index>(*d.tt_, *rtt);
            auto elevators =
                c.elevators_.has_value()
                    ? co_await update_elevators(*c.elevators_, d, *rtt)
                    : std::move(d.rt_->e_);
            d.rt_ = std::make_shared<rt>(std::move(rtt), std::move(elevators),
                                         std::move(railviz_rt));
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
