#include <string_view>

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

namespace fs = std::filesystem;

namespace motis {

int server(data d, config const& c, std::string_view const motis_version) {
  auto scheduler = ctx::scheduler<ctx_data>{};
  auto m = motis_instance{ctx_exec{scheduler.runner_.ios(), scheduler}, d, c,
                          motis_version};

  auto lbs = std::vector<net::lb>{};
  if (c.server_.value_or(config::server{}).lbs_) {
    lbs = utl::to_vec(*c.server_.value_or(config::server{}).lbs_,
                      [&](std::string const& url) {
                        return net::lb{scheduler.runner_.ios(), url, m.qr_};
                      });
  }

  auto s = net::web_server{scheduler.runner_.ios()};
  s.set_timeout(std::chrono::minutes{5});
  s.on_http_request(m.qr_);

  auto ec = boost::system::error_code{};
  auto const server_config = c.server_.value_or(config::server{});
  s.init(server_config.host_, server_config.port_, ec);
  if (ec) {
    std::cerr << "error: " << ec << "\n";
    return 1;
  }

  auto const stop = net::stop_handler(scheduler.runner_.ios(), [&]() {
    utl::log_info("motis.server", "shutdown");
    for (auto& lb : lbs) {
      lb.stop();
    }
    s.stop();
    m.stop();
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
  m.run(d, c);
  scheduler.runner_.run(c.n_threads());
  m.join();

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
