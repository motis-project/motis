#include "motis/osm_rt/osm_rt.h"

#include <algorithm>
#include <string_view>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"

#include "utl/read_file.h"
#include "utl/verify.h"

#include "geo/box.h"
#include "geo/latlng.h"

#include "nigiri/logging.h"

#include "osr/lookup.h"
#include "osr/ways.h"

#include "osm-rt.pb.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/http_req.h"
#include "motis/repeat.h"

namespace n = nigiri;
namespace asio = boost::asio;
using asio::awaitable;

namespace motis {

// Feed geometry is generated from OSM data, so its points coincide with OSM
// node positions. Tolerance only covers coordinate rounding / minor drift
// between the feed's and the loaded OSM snapshot.
constexpr auto const kMaxMatchDistance = 3.0;  // meters

namespace {

bool is_access_blocking(std::string_view const key,
                        std::string_view const val) {
  return (val == "no" || val == "private") &&
         (key == "access" || key == "vehicle" || key == "motor_vehicle" ||
          key == "motorcar");
}

}  // namespace

osr::bitvec<osr::node_idx_t> osm_rt_blocked_nodes(
    osr::ways const& w,
    osr::lookup const& l,
    std::vector<std::string> const& feed_messages) {
  auto blocked = osr::bitvec<osr::node_idx_t>{};
  blocked.resize(w.n_nodes());

  auto n_entities = 0U;
  auto n_blocking = 0U;
  auto n_unmatched = 0U;
  auto n_nodes = 0U;

  auto const match_pt = [&](geo::latlng const& pos, auto&& fn) {
    l.find(geo::box{pos, kMaxMatchDistance}, [&](osr::way_idx_t const way) {
      for (auto const node : w.r_->way_nodes_[way]) {
        if (geo::distance(w.get_node_pos(node).as_latlng(), pos) <=
            kMaxMatchDistance) {
          fn(node);
        }
      }
    });
  };

  for (auto const& body : feed_messages) {
    if (body.empty()) {
      continue;
    }

    auto msg = osm_rt::FeedMessage{};
    utl::verify(msg.ParseFromString(body), "OSM-RT: invalid feed message");
    utl::verify(msg.header().compat_version() <= 1U,
                "OSM-RT: unsupported compat_version={}",
                msg.header().compat_version());

    auto const str = [&](std::uint32_t const i) {
      return i < static_cast<std::uint32_t>(msg.stringtable().s_size())
                 ? std::string_view{msg.stringtable().s(static_cast<int>(i))}
                 : std::string_view{};
    };

    for (auto const& e : msg.entity()) {
      ++n_entities;

      auto is_blocking = false;
      for (auto i = 0; i != std::min(e.keys_size(), e.vals_size()); ++i) {
        if (is_access_blocking(str(e.keys(i)), str(e.vals(i)))) {
          is_blocking = true;
          break;
        }
      }
      if (!is_blocking) {
        continue;
      }
      ++n_blocking;

      auto pts = std::vector<geo::latlng>{};
      pts.reserve(static_cast<std::size_t>(e.lat_size()));
      auto lat = std::int64_t{0};
      auto lon = std::int64_t{0};
      for (auto i = 0; i != std::min(e.lat_size(), e.lon_size()); ++i) {
        lat += e.lat(i);
        lon += e.lon(i);
        pts.emplace_back(static_cast<double>(lat) * 1e-7,
                         static_cast<double>(lon) * 1e-7);
      }

      auto matched = false;
      auto const block = [&](osr::node_idx_t const node) {
        if (!blocked.test(node)) {
          ++n_nodes;
        }
        blocked.set(node, true);
        matched = true;
      };

      // Prefer blocking only interior nodes: this closes the referenced way
      // but keeps the junctions at both ends usable for crossing traffic.
      if (pts.size() > 2U) {
        for (auto i = std::size_t{1U}; i != pts.size() - 1U; ++i) {
          match_pt(pts[i], block);
        }
      }
      if (!matched) {  // short way without interior nodes
        for (auto const& p : pts) {
          match_pt(p, block);
        }
      }
      if (!matched) {
        ++n_unmatched;
      }
    }
  }

  n::log(n::log_lvl::info, "motis.osmrt",
         "OSM-RT update: {} entities, {} access-blocking, {} unmatched, {} "
         "blocked nodes",
         n_entities, n_blocking, n_unmatched, n_nodes);

  return blocked;
}

void run_osm_rt_update(asio::io_context& ioc, config const& c, data& d) {
  asio::co_spawn(
      ioc,
      [&c, &d]() -> awaitable<void> {
        auto const sr = *c.get_street_routing();
        auto bodies = std::vector<std::string>(sr.osm_rt_.size());
        co_await repeat(
            std::chrono::seconds{sr.osm_rt_update_interval_}, "osm-rt update",
            [&]() -> awaitable<void> {
              auto i = 0U;
              for (auto const& [tag, feed] : sr.osm_rt_) {
                try {
                  if (feed.url_.starts_with("http://") ||
                      feed.url_.starts_with("https://")) {
                    auto const res = co_await http_GET(
                        boost::urls::url{feed.url_},
                        feed.headers_.value_or(headers_t{}),
                        std::chrono::seconds{sr.osm_rt_http_timeout_});
                    bodies[i] = get_http_body(res);
                  } else {
                    auto const body = utl::read_file(feed.url_.c_str());
                    utl::verify(body.has_value(),
                                "could not read OSM-RT feed from {}",
                                feed.url_);
                    bodies[i] = *body;
                  }
                } catch (std::exception const& e) {
                  // keep the last successfully fetched body for this feed
                  n::log(n::log_lvl::error, "motis.osmrt",
                         "OSM-RT FETCH ERROR: tag={}, url={}, error={}", tag,
                         feed.url_, e.what());
                }
                ++i;
              }

              auto snapshot = std::make_shared<osm_rt_data>(
                  osm_rt_data{osm_rt_blocked_nodes(*d.w_, *d.l_, bodies)});
              std::atomic_store(&d.osm_rt_, std::move(snapshot));
            });
      },
      asio::detached);
}

}  // namespace motis
