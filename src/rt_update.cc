#include "motis/rt_update.h"

#include "utl/timer.h"

#include "nigiri/rt/create_rt_timetable.h"
#include "nigiri/rt/gtfsrt_update.h"
#include "nigiri/rt/rt_timetable.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/http_req.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

awaitable<void> rt_update(config const& c,
                          nigiri::timetable const& tt,
                          tag_lookup const& tags,
                          std::shared_ptr<rt>& r) {
  auto const t = utl::scoped_timer{"rt_update"};

  auto const no_hdr = headers_t{};
  auto gtfs_rt = std::vector<std::tuple<n::source_idx_t, boost::urls::url,
                                        awaitable<http_response>>>{};
  for (auto const& [tag, d] : c.timetable_->datasets_) {
    if (!d.rt_.has_value()) {
      continue;
    }

    auto const src = tags.get_src(tag);
    for (auto const& ep : *d.rt_) {
      auto const url = boost::urls::url{ep.url_};
      gtfs_rt.emplace_back(
          src, url,
          http_GET(url, ep.headers_.has_value() ? *ep.headers_ : no_hdr));
    }
  }

  auto const today = std::chrono::time_point_cast<date::days>(
      std::chrono::system_clock::now());
  auto rtt = std::make_unique<n::rt_timetable>(
      c.timetable_->incremental_rt_update_
          ? n::rt_timetable{*r->rtt_}
          : n::rt::create_rt_timetable(tt, today));

  auto statistics = std::vector<n::rt::statistics>{};
  for (auto& [src, url, response] : gtfs_rt) {
    // alternatively: make_parallel_group
    auto const res = co_await std::move(response);
    auto const stats = n::rt::gtfsrt_update_buf(
        tt, *rtt, src, tags.get_tag(src),
        boost::beast::buffers_to_string(res.body().data()));
    statistics.emplace_back(stats);
  }

  for (auto const [endpoint, stats] : utl::zip(gtfs_rt, statistics)) {
    auto const& [src, url, response] = endpoint;
    fmt::println("rt update stats for {}: {}", fmt::streamed(url),
                 fmt::streamed(stats));
  }

  r = std::make_shared<rt>(std::move(rtt), std::move(r->e_));

  co_return;
}

}  // namespace motis