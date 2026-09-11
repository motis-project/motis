#include <functional>
#include <memory>
#include <string_view>
#include <thread>

#include "boost/asio/io_context.hpp"

#include "fmt/format.h"

#include "net/lb.h"
#include "net/run.h"
#include "net/stop_handler.h"
#include "net/web_server/web_server.h"

#include "utl/enumerate.h"
#include "utl/init_from.h"
#include "utl/logging.h"
#include "utl/parser/arg_parser.h"

#include "ctx/ctx.h"

#include "motis/config.h"
#include "motis/ctx_data.h"
#include "motis/ctx_exec.h"
#include "motis/data.h"
#include "motis/motis_instance.h"
#include "motis/static_reload.h"

namespace fs = std::filesystem;

namespace motis {

namespace {

using instance_t = motis_instance<ctx_exec>;

// One "generation" of served state: the data loaded from disk (static +
// initial rt timetable, osr graph, ...) plus the motis_instance (registered
// HTTP routes, each bound to this exact `data`) built from it.
//
// Hot-reloading the static timetable builds a new `served` off to the side
// (see static_reload.h) and atomically swaps `server`'s `current_` pointer
// to it. `dispatcher` (below) pins whichever generation was current when a
// request came in for the full lifetime of that request, so a generation
// is only ever destroyed once every request it served has been answered
// and its background rt/gbfs update threads have been joined.
struct served {
  served(std::unique_ptr<data> d, std::unique_ptr<instance_t> i)
      : data_{std::move(d)}, instance_{std::move(i)} {}
  std::unique_ptr<data> data_;
  std::unique_ptr<instance_t> instance_;
};

// Forwards every request to whichever `served` generation is current at the
// moment it comes in, and keeps that generation alive for as long as the
// request takes to answer, even if a reload swaps in a new generation
// while it's still in flight.
struct dispatcher {
  void operator()(net::web_server::http_req_t req,
                  net::web_server::http_res_cb_t cb,
                  bool const is_ssl) const {
    auto inst = std::atomic_load(current_);
    (*inst->instance_)(
        std::move(req),
        [inst, cb = std::move(cb)](net::web_server::http_res_t&& res) mutable {
          cb(std::move(res));
        },
        is_ssl);
  }

  std::shared_ptr<served> const* current_;
};

// Stops a superseded generation's background update threads and joins them
// on a throwaway thread so the reload path never blocks on it; the
// generation itself (and the `data` it owns: mmapped osr/tt files, etc.)
// is only actually freed once this join finishes *and* every in-flight
// request that was pinning it (via `dispatcher`) has finished.
void retire(std::shared_ptr<served> old_served) {
  old_served->instance_->stop();
  std::thread{[old = std::move(old_served)]() mutable {
    old->instance_->join();
    old.reset();
  }}.detach();
}

}  // namespace

int server(data d, config const& c, std::string_view const motis_version) {
  auto scheduler = ctx::scheduler<ctx_data>{};
  auto const data_path = d.path_;

  auto initial_data = std::make_unique<data>(std::move(d));
  auto initial_instance =
      std::make_unique<instance_t>(ctx_exec{scheduler.runner_.ios(), scheduler},
                                   *initial_data, c, motis_version);

  auto current = std::shared_ptr<served>{std::make_shared<served>(
      std::move(initial_data), std::move(initial_instance))};

  auto const dispatch = dispatcher{&current};

  auto lbs = std::vector<net::lb>{};
  if (c.server_.value_or(config::server{}).lbs_) {
    lbs = utl::to_vec(*c.server_.value_or(config::server{}).lbs_,
                      [&](std::string const& url) {
                        return net::lb{scheduler.runner_.ios(), url, dispatch};
                      });
  }

  auto s = net::web_server{scheduler.runner_.ios()};
  s.set_timeout(std::chrono::minutes{5});
  s.on_http_request(dispatch);

  auto ec = boost::system::error_code{};
  auto const server_config = c.server_.value_or(config::server{});
  s.init(server_config.host_, server_config.port_, ec);
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  // Runs the (rare, potentially slow: download + full timetable rebuild)
  // static reload on its own OS thread, like the rt/gbfs updaters, never
  // on the shared HTTP request-handling thread pool, so a reload never
  // steals capacity from concurrent request processing.
  auto reload_thread = io_thread{};
  if (c.requires_static_reload()) {
    reload_thread = io_thread{
        "motis static reload", [&](boost::asio::io_context& ioc) {
          run_static_reload(ioc, c, data_path, [&]() {
            auto new_data = std::make_unique<data>(data_path, c);
            auto new_instance = std::make_unique<instance_t>(
                ctx_exec{scheduler.runner_.ios(), scheduler}, *new_data, c,
                motis_version);
            new_instance->run(*new_data, c);

            auto new_served = std::make_shared<served>(std::move(new_data),
                                                       std::move(new_instance));
            retire(std::atomic_exchange(&current, std::move(new_served)));
          });
        }};
  }

  auto const stop = net::stop_handler(scheduler.runner_.ios(), [&]() {
    utl::log_info("motis.server", "shutdown");
    for (auto& lb : lbs) {
      lb.stop();
    }
    s.stop();
    reload_thread.stop();
    std::atomic_load(&current)->instance_->stop();
    scheduler.runner_.stop();
  });

  utl::log_info(
      "motis.server",
      "n_threads={}, listening on {}:{}\nlocal link: http://localhost:{}",
      c.n_threads(), server_config.host_, server_config.port_,
      server_config.port_);

  for (auto& lb : lbs) {
    lb.run();
  }
  s.run();
  // safe to access `current` directly here (not via atomic_load): no other
  // thread can be reading/writing it yet, the reload thread only touches
  // `current` from within the `run_static_reload` callback above, and the
  // scheduler (which would let request-handling threads observe `current`)
  // isn't running yet either.
  current->instance_->run(*current->data_, c);

  scheduler.runner_.run(c.n_threads());
  reload_thread.join();
  std::atomic_load(&current)->instance_->join();

  return 0;
}

unsigned get_api_version(boost::urls::url_view const& url) {
  if (url.encoded_path().length() > 7) {
    return utl::parse<unsigned>(
        std::string_view{url.encoded_path().substr(6, 2)});
  }
  return 0U;
}

}  // namespace motis
