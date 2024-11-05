#include "motis/gbfs/update.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>

#include <iostream>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"

#include "boost/json.hpp"

#include "openssl/sha.h"

#include "fmt/format.h"

#include "utl/helpers/algorithm.h"
#include "utl/timer.h"
#include "utl/to_vec.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/gbfs/data.h"
#include "motis/http_client.h"
#include "motis/http_req.h"

#include "motis/gbfs/osr_mapping.h"
#include "motis/gbfs/parser.h"

namespace asio = boost::asio;
using asio::awaitable;

namespace json = boost::json;

namespace motis::gbfs {

struct gbfs_file {
  json::value json_;
  std::string hash_;
};

std::string read_file(std::filesystem::path const& path) {
  auto is = std::ifstream{path};
  auto buf = std::stringstream{};
  buf << is.rdbuf();
  return buf.str();
}

std::string hash(std::string_view const s) {
  unsigned char hash[SHA256_DIGEST_LENGTH];
  SHA256(reinterpret_cast<unsigned char const*>(s.data()), s.size(), hash);
  return std::string{reinterpret_cast<char*>(hash), SHA256_DIGEST_LENGTH};
}

awaitable<gbfs_file> fetch_file(std::string_view const name,
                                std::string_view const url,
                                headers_t const& headers,
                                std::optional<std::filesystem::path> const& dir,
                                http_client& client) {
  auto content = std::string{};
  if (dir.has_value()) {
    content = read_file(*dir / fmt::format("{}.json", name));
  } else {
    auto const res = co_await client.get(boost::urls::url{url}, headers);
    content = get_http_body(res);
  }
  co_return gbfs_file{.json_ = json::parse(content), .hash_ = hash(content)};
}

void add_provider_to_rtree(gbfs_data& data, gbfs_provider const& provider) {
  for (auto const& station : provider.stations_) {
    data.provider_rtree_.add(station.second.info_.pos_, provider.idx_);
  }
  for (auto const& vehicle : provider.vehicle_status_) {
    if (vehicle.station_id_.empty()) {
      data.provider_rtree_.add(vehicle.pos_, provider.idx_);
    }
  }
}

awaitable<void> load_feed(config::gbfs const& c,
                          osr::ways const& w,
                          osr::lookup const& l,
                          gbfs_data* d,
                          std::string const& prefix,
                          std::string const& id,
                          std::string const& url,
                          headers_t const& headers,
                          std::optional<std::filesystem::path> const& dir,
                          http_client& client);

struct manifest_feed {
  std::string combined_id_{};
  std::string url_{};
};

awaitable<void> load_manifest(config::gbfs const& c,
                              osr::ways const& w,
                              osr::lookup const& l,
                              gbfs_data* d,
                              std::string const& prefix,
                              headers_t const& headers,
                              http_client& client,
                              boost::json::object const& root) {
  auto feeds = std::vector<manifest_feed>{};
  if (root.contains("data") &&
      root.at("data").as_object().contains("datasets")) {
    // GBFS 3.x manifest.json
    for (auto const& dataset : root.at("data").at("datasets").as_array()) {
      auto const system_id =
          static_cast<std::string>(dataset.at("system_id").as_string());
      auto const combined_id = fmt::format("{}:{}", prefix, system_id);

      auto const& versions = dataset.at("versions").as_array();
      if (versions.empty()) {
        continue;
      }
      // versions array must be sorted by increasing version number
      auto const& latest_version = versions.back().as_object();
      feeds.emplace_back(manifest_feed{
          combined_id,
          static_cast<std::string>(latest_version.at("url").as_string())});
    }
  } else if (root.contains("systems")) {
    // Lamassu 2.3 format
    for (auto const& system : root.at("systems").as_array()) {
      auto const system_id =
          static_cast<std::string>(system.at("id").as_string());
      auto const combined_id = fmt::format("{}:{}", prefix, system_id);
      feeds.emplace_back(manifest_feed{
          combined_id, static_cast<std::string>(system.at("url").as_string())});
    }
  }

  auto executor = co_await asio::this_coro::executor;
  auto awaitables = utl::to_vec(feeds, [&](auto const& feed) {
    return boost::asio::co_spawn(
        executor,
        [&, feed]() -> awaitable<void> {
          co_await load_feed(c, w, l, d, prefix, feed.combined_id_, feed.url_,
                             headers, {}, client);
        },
        asio::deferred);
  });

  auto x =
      co_await asio::experimental::make_parallel_group(awaitables)
          .async_wait(asio::experimental::wait_for_all(), asio::use_awaitable);
}

awaitable<void> load_feed(config::gbfs const& c,
                          osr::ways const& w,
                          osr::lookup const& l,
                          gbfs_data* d,
                          std::string const& prefix,
                          std::string const& id,
                          std::string const& url,
                          headers_t const& headers,
                          std::optional<std::filesystem::path> const& dir,
                          http_client& client) {
  std::cout << "[GBFS] loading feed " << id << ": " << url << std::endl;
  try {
    auto const discovery =
        co_await fetch_file("gbfs", url, headers, dir, client);

    auto const& root = discovery.json_.as_object();
    if ((root.contains("data") &&
         root.at("data").as_object().contains("datasets")) ||
        root.contains("systems")) {
      // File is not an individual feed, but a manifest.json / Lamassu file
      co_return co_await load_manifest(c, w, l, d, id, headers, client, root);
    }

    auto const urls = parse_discovery(discovery.json_);

    auto const fetch = [&](std::string_view const name) {
      return fetch_file(name, urls.at(name), headers, dir, client);
    };

    auto const provider_idx = gbfs_provider_idx_t{d->providers_.size()};
    d->provider_by_id_[id] = provider_idx;
    auto provider =
        d->providers_.emplace_back(std::make_unique<gbfs_provider>()).get();
    provider->id_ = id;
    provider->idx_ = provider_idx;

    auto const set_default_restrictions =
        [&](config::gbfs::restrictions const& r) {
          provider->default_restrictions_ = geofencing_restrictions{
              .ride_start_allowed_ = r.ride_start_allowed_,
              .ride_end_allowed_ = r.ride_end_allowed_,
              .ride_through_allowed_ = r.ride_through_allowed_};
        };

    if (auto const it = c.default_restrictions_.find(id);
        it != end(c.default_restrictions_)) {
      set_default_restrictions(it->second);
    } else if (auto const prefix_it = c.default_restrictions_.find(prefix);
               prefix_it != end(c.default_restrictions_)) {
      set_default_restrictions(prefix_it->second);
    }

    if (urls.contains("system_information")) {
      load_system_information(*provider,
                              (co_await fetch("system_information")).json_);
    }

    if (urls.contains("vehicle_types")) {
      load_vehicle_types(*provider, (co_await fetch("vehicle_types")).json_);
    }

    if (urls.contains("station_information") &&
        urls.contains("station_status")) {
      load_station_information(*provider,
                               (co_await fetch("station_information")).json_);
      load_station_status(*provider, (co_await fetch("station_status")).json_);
    }

    if (urls.contains("vehicle_status")) {  // 3.x
      load_vehicle_status(*provider, (co_await fetch("vehicle_status")).json_);
    } else if (urls.contains("free_bike_status")) {  // 2.x
      load_vehicle_status(*provider,
                          (co_await fetch("free_bike_status")).json_);
    }

    if (urls.contains("geofencing_zones")) {
      load_geofencing_zones(*provider,
                            (co_await fetch("geofencing_zones")).json_);
    }

    map_geofencing_zones(w, l, *provider);
    map_stations(w, l, *provider);
    map_vehicles(w, l, *provider);

    provider->has_vehicles_to_rent_ = provider->start_allowed_.count() != 0;

    if (provider->has_vehicles_to_rent_) {
      add_provider_to_rtree(*d, *provider);
    }

    std::cout << "[GBFS] provider " << id
              << " initialized: " << provider->stations_.size() << " stations ("
              << utl::count_if(
                     provider->stations_,
                     [](auto const& s) {
                       return s.second.status_.is_renting_ &&
                              s.second.status_.num_vehicles_available_ > 0;
                     })
              << " renting, "
              << utl::count_if(provider->stations_,
                               [](auto const& s) {
                                 return s.second.status_.is_returning_;
                               })
              << " returning), " << provider->vehicle_status_.size()
              << " vehicles, " << provider->vehicle_types_.size()
              << " vehicle types, " << provider->geofencing_zones_.zones_.size()
              << " geofencing zones\n";
  } catch (std::exception const& ex) {
    std::cerr << "[GBFS] error loading feed " << id << " (" << url
              << "): " << ex.what() << "\n";
  }
}

awaitable<void> update(config const& c,
                       osr::ways const& w,
                       osr::lookup const& l,
                       std::shared_ptr<gbfs_data>& data_ptr) {
  auto const t = utl::scoped_timer{"gbfs::update"};

  if (!c.gbfs_.has_value()) {
    co_return;
  }

  auto d = std::make_shared<gbfs_data>();
  auto client = http_client{};
  auto const no_hdr = headers_t{};
  client.timeout_ = std::chrono::seconds{c.gbfs_->http_timeout_};

  auto executor = co_await asio::this_coro::executor;
  auto awaitables = utl::to_vec(c.gbfs_->feeds_, [&](auto const& f) {
    auto const& id = f.first;
    auto const& feed = f.second;
    auto const dir =
        feed.url_.starts_with("http:") || feed.url_.starts_with("https:")
            ? std::nullopt
            : std::optional<std::filesystem::path>{feed.url_};

    return boost::asio::co_spawn(
        executor,
        [id, feed, dir, &c, &d, &w, &l, &no_hdr, &client]() -> awaitable<void> {
          co_await load_feed(c.gbfs_.value(), w, l, d.get(), "", id, feed.url_,
                             feed.headers_.value_or(no_hdr), dir, client);
        },
        asio::deferred);
  });

  auto x =
      co_await asio::experimental::make_parallel_group(awaitables)
          .async_wait(asio::experimental::wait_for_all(), asio::use_awaitable);

  data_ptr = d;
  co_return;
}

void run_gbfs_update(boost::asio::io_context& ioc,
                     config const& c,
                     osr::ways const& w,
                     osr::lookup const& l,
                     std::shared_ptr<gbfs_data>& data_ptr) {
  boost::asio::co_spawn(
      ioc,
      [&]() -> awaitable<void> {
        auto executor = co_await asio::this_coro::executor;
        auto timer = asio::steady_timer{executor};
        auto ec = boost::system::error_code{};

        while (true) {
          // Remember when we started so we can schedule the next update.
          auto const start = std::chrono::steady_clock::now();

          co_await update(c, w, l, data_ptr);

          // Schedule next update.
          timer.expires_at(start +
                           std::chrono::seconds{c.gbfs_->update_interval_});
          co_await timer.async_wait(
              asio::redirect_error(asio::use_awaitable, ec));
          if (ec == asio::error::operation_aborted) {
            co_return;
          }
        }
      },
      boost::asio::detached);
}

}  // namespace motis::gbfs
