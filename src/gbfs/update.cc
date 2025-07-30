#include "motis/gbfs/update.h"

#include <cassert>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string_view>
#include <utility>

#include <iostream>

#include "boost/asio/co_spawn.hpp"
#include "boost/asio/detached.hpp"
#include "boost/asio/experimental/awaitable_operators.hpp"
#include "boost/asio/experimental/parallel_group.hpp"
#include "boost/asio/redirect_error.hpp"
#include "boost/asio/steady_timer.hpp"

#include "boost/json.hpp"

#include "cista/hash.h"

#include "fmt/format.h"

#include "utl/helpers/algorithm.h"
#include "utl/overloaded.h"
#include "utl/sorted_diff.h"
#include "utl/timer.h"
#include "utl/to_vec.h"

#include "motis/config.h"
#include "motis/data.h"
#include "motis/gbfs/data.h"
#include "motis/http_client.h"
#include "motis/http_req.h"

#include "motis/gbfs/compression.h"
#include "motis/gbfs/osr_mapping.h"
#include "motis/gbfs/parser.h"
#include "motis/gbfs/partition.h"
#include "motis/gbfs/routing_data.h"

namespace asio = boost::asio;
using asio::awaitable;
using namespace asio::experimental::awaitable_operators;

namespace json = boost::json;

namespace motis::gbfs {

struct gbfs_file {
  json::value json_;
  cista::hash_t hash_{};
  std::chrono::system_clock::time_point next_refresh_;
};

std::string read_file(std::filesystem::path const& path) {
  auto is = std::ifstream{path};
  auto buf = std::stringstream{};
  buf << is.rdbuf();
  return buf.str();
}

bool needs_refresh(file_info const& fi) {
  return fi.needs_update(std::chrono::system_clock::now());
}

// try to hash only the value of the "data" key to ignore fields like
// "last_updated"
cista::hash_t hash_gbfs_data(std::string_view const json) {
  auto const pos = json.find("\"data\"");
  if (pos == std::string_view::npos) {
    return cista::hash(json);
  }

  auto i = pos + 6;
  auto const skip_whitespace = [&]() {
    while (i < json.size() && (json[i] == ' ' || json[i] == '\n' ||
                               json[i] == '\r' || json[i] == '\t')) {
      ++i;
    }
  };
  skip_whitespace();

  if (i >= json.size() || json[i++] != ':') {
    return cista::hash(json);
  }

  skip_whitespace();

  if (i >= json.size() || json[i] != '{') {
    return cista::hash(json);
  }

  auto const start = i;
  auto depth = 1;
  auto in_string = false;

  while (++i < json.size()) {
    if (in_string) {
      if (json[i] == '"' && json[i - 1] != '\\') {
        in_string = false;
      }
      continue;
    }

    switch (json[i]) {
      case '"': in_string = true; break;
      case '{': ++depth; break;
      case '}':
        if (--depth == 0) {
          return cista::hash(json.substr(start, i - start + 1));
        }
    }
  }

  return cista::hash(json);
}

std::chrono::system_clock::time_point get_expiry(
    boost::json::object const& root,
    std::chrono::seconds const def = std::chrono::seconds{0}) {
  auto const now = std::chrono::system_clock::now();
  if (root.contains("data")) {
    auto const& data = root.at("data").as_object();
    if (data.contains("ttl")) {
      return now + std::chrono::seconds{data.at("ttl").to_number<int>()};
    }
  }
  return now + def;
}

struct gbfs_update {
  gbfs_update(config::gbfs const& c,
              osr::ways const& w,
              osr::lookup const& l,
              gbfs_data* d,
              gbfs_data const* prev_d)
      : c_{c}, w_{w}, l_{l}, d_{d}, prev_d_{prev_d} {
    client_.timeout_ = std::chrono::seconds{c.http_timeout_};
    if (c.proxy_ && !c.proxy_->empty()) {
      client_.set_proxy(boost::urls::url{*c.proxy_});
    }
  }

  awaitable<void> run() {
    auto executor = co_await asio::this_coro::executor;
    if (prev_d_ == nullptr) {
      // this is first time gbfs_update is run: initialize feeds from config
      d_->aggregated_feeds_ =
          std::make_shared<std::vector<std::unique_ptr<aggregated_feed>>>();
      d_->standalone_feeds_ =
          std::make_shared<std::vector<std::unique_ptr<provider_feed>>>();

      auto const no_hdr = headers_t{};
      auto awaitables = utl::to_vec(c_.feeds_, [&](auto const& f) {
        auto const& id = f.first;
        auto const& feed = f.second;
        auto const dir =
            feed.url_.starts_with("http:") || feed.url_.starts_with("https:")
                ? std::nullopt
                : std::optional<std::filesystem::path>{feed.url_};

        return boost::asio::co_spawn(
            executor,
            [this, id, feed, dir, &no_hdr]() -> awaitable<void> {
              co_await init_feed(id, feed.url_, feed.headers_.value_or(no_hdr),
                                 dir);
            },
            asio::deferred);
      });

      co_await asio::experimental::make_parallel_group(awaitables)
          .async_wait(asio::experimental::wait_for_all(), asio::use_awaitable);
    } else {
      // update run: copy over data from previous state and update feeds
      // where necessary
      d_->aggregated_feeds_ = prev_d_->aggregated_feeds_;
      d_->standalone_feeds_ = prev_d_->standalone_feeds_;
      // the set of providers can change if aggregated feeds are used + change.
      // gbfs_provider_idx_t for existing providers is stable, if a provider is
      // removed its entry is set to a nullptr. new providers may be added.
      d_->providers_.resize(prev_d_->providers_.size());
      d_->provider_by_id_ = prev_d_->provider_by_id_;
      d_->provider_rtree_ = prev_d_->provider_rtree_;
      d_->cache_ = prev_d_->cache_;

      if (!d_->aggregated_feeds_->empty()) {
        co_await asio::experimental::make_parallel_group(
            utl::to_vec(*d_->aggregated_feeds_,
                        [&](auto const& af) {
                          return boost::asio::co_spawn(
                              executor,
                              [this, af = af.get()]() -> awaitable<void> {
                                co_await update_aggregated_feed(*af);
                              },
                              asio::deferred);
                        }))
            .async_wait(asio::experimental::wait_for_all(),
                        asio::use_awaitable);
      }

      if (!d_->standalone_feeds_->empty()) {
        co_await asio::experimental::make_parallel_group(
            utl::to_vec(*d_->standalone_feeds_,
                        [&](auto const& pf) {
                          return boost::asio::co_spawn(
                              executor,
                              [this, pf = pf.get()]() -> awaitable<void> {
                                co_await update_provider_feed(*pf);
                              },
                              asio::deferred);
                        }))
            .async_wait(asio::experimental::wait_for_all(),
                        asio::use_awaitable);
      }
    }
  }

  awaitable<void> init_feed(std::string const& id,
                            std::string const& url,
                            headers_t const& headers,
                            std::optional<std::filesystem::path> const& dir) {
    // initialization of a (standalone or aggregated) feed from the config
    try {
      auto discovery = co_await fetch_file("gbfs", url, headers, dir);
      auto const& root = discovery.json_.as_object();
      if ((root.contains("data") &&
           root.at("data").as_object().contains("datasets")) ||
          root.contains("systems")) {
        // file is not an individual feed, but a manifest.json / Lamassu file
        co_return co_await init_aggregated_feed(id, url, headers, root);
      }

      auto saf =
          d_->standalone_feeds_
              ->emplace_back(std::make_unique<provider_feed>(provider_feed{
                  .id_ = id,
                  .url_ = url,
                  .headers_ = headers,
                  .dir_ = dir,
                  .default_restrictions_ = lookup_default_restrictions("", id),
                  .default_return_constraint_ =
                      lookup_default_return_constraint("", id)}))
              .get();

      co_return co_await update_provider_feed(*saf, std::move(discovery));
    } catch (std::exception const& ex) {
      std::cerr << "[GBFS] error initializing feed " << id << " (" << url
                << "): " << ex.what() << "\n";
    }
  }

  awaitable<void> update_provider_feed(
      provider_feed const& pf,
      std::optional<gbfs_file> discovery = std::nullopt) {
    auto& provider = add_provider(pf);

    // check if exists in old data - if so, reuse existing file infos
    gbfs_provider const* prev_provider = nullptr;
    if (prev_d_ != nullptr) {
      if (auto const it = prev_d_->provider_by_id_.find(pf.id_);
          it != end(prev_d_->provider_by_id_)) {
        prev_provider = prev_d_->providers_[it->second].get();
        if (prev_provider != nullptr) {
          provider.file_infos_ = prev_provider->file_infos_;
        }
      }
    }
    if (!provider.file_infos_) {
      provider.file_infos_ = std::make_shared<provider_file_infos>();
    }

    co_return co_await process_provider_feed(pf, provider, prev_provider,
                                             std::move(discovery));
  }

  gbfs_provider& add_provider(provider_feed const& pf) {
    auto const init_provider = [&](gbfs_provider& provider,
                                   gbfs_provider_idx_t const idx) {
      provider.id_ = pf.id_;
      provider.idx_ = idx;
      provider.default_restrictions_ = pf.default_restrictions_;
      provider.default_return_constraint_ = pf.default_return_constraint_;
    };

    if (auto it = d_->provider_by_id_.find(pf.id_);
        it != end(d_->provider_by_id_)) {
      // existing provider, keep idx
      auto const idx = it->second;
      assert(d_->providers_.at(idx) == nullptr);
      d_->providers_[idx] = std::make_unique<gbfs_provider>();
      auto& provider = *d_->providers_[idx].get();
      init_provider(provider, idx);
      return provider;
    } else {
      // new provider
      auto const idx = gbfs_provider_idx_t{d_->providers_.size()};
      auto& provider =
          *d_->providers_.emplace_back(std::make_unique<gbfs_provider>()).get();
      d_->provider_by_id_[pf.id_] = idx;
      init_provider(provider, idx);
      return provider;
    }
  }

  awaitable<void> process_provider_feed(
      provider_feed const& pf,
      gbfs_provider& provider,
      gbfs_provider const* prev_provider,
      std::optional<gbfs_file> discovery = std::nullopt) {
    auto& file_infos = provider.file_infos_;
    auto data_changed = false;

    try {
      if (!discovery && needs_refresh(provider.file_infos_->urls_fi_)) {
        discovery = co_await fetch_file("gbfs", pf.url_, pf.headers_, pf.dir_);
      }
      if (discovery) {
        file_infos->urls_ = parse_discovery(discovery->json_);
        file_infos->urls_fi_.expiry_ = discovery->next_refresh_;
        file_infos->urls_fi_.hash_ = discovery->hash_;
      }

      auto const update = [&](std::string_view const name, file_info& fi,
                              auto const& fn,
                              bool const force = false) -> awaitable<bool> {
        if (force || (file_infos->urls_.contains(name) && needs_refresh(fi))) {
          auto file = co_await fetch_file(name, file_infos->urls_.at(name),
                                          pf.headers_, pf.dir_);
          auto const hash_changed = file.hash_ != fi.hash_;
          auto j_root = file.json_.as_object();
          fi.expiry_ = file.next_refresh_;
          fi.hash_ = file.hash_;
          fn(provider, file.json_);
          co_return hash_changed;
        }
        co_return false;
      };

      auto const sys_info_updated = co_await update(
          "system_information", file_infos->system_information_fi_,
          load_system_information);
      if (!sys_info_updated && prev_provider != nullptr) {
        provider.sys_info_ = prev_provider->sys_info_;
      }

      auto const vehicle_types_updated = co_await update(
          "vehicle_types", file_infos->vehicle_types_fi_, load_vehicle_types);
      if (!vehicle_types_updated && prev_provider != nullptr) {
        provider.vehicle_types_ = prev_provider->vehicle_types_;
        provider.vehicle_types_map_ = prev_provider->vehicle_types_map_;
        provider.temp_vehicle_types_ = prev_provider->temp_vehicle_types_;
      }

      auto const stations_updated = co_await update(
          "station_information", file_infos->station_information_fi_,
          load_station_information);
      if (!stations_updated && prev_provider != nullptr) {
        provider.stations_ = prev_provider->stations_;
      }

      auto const station_status_updated =
          co_await update("station_status", file_infos->station_status_fi_,
                          load_station_status, stations_updated);

      auto const vehicle_status_updated =
          co_await update("vehicle_status", file_infos->vehicle_status_fi_,
                          load_vehicle_status)  // 3.x
          || co_await update("free_bike_status", file_infos->vehicle_status_fi_,
                             load_vehicle_status);  // 1.x / 2.x
      if (!vehicle_status_updated && prev_provider != nullptr) {
        provider.vehicle_status_ = prev_provider->vehicle_status_;
      }

      auto const geofencing_updated =
          co_await update("geofencing_zones", file_infos->geofencing_zones_fi_,
                          load_geofencing_zones);
      if (!geofencing_updated && prev_provider != nullptr) {
        provider.geofencing_zones_ = prev_provider->geofencing_zones_;
      }

      if (prev_provider != nullptr) {
        provider.has_vehicles_to_rent_ = prev_provider->has_vehicles_to_rent_;
      }

      data_changed = vehicle_types_updated || stations_updated ||
                     station_status_updated || vehicle_status_updated ||
                     geofencing_updated;
    } catch (std::exception const& ex) {
      std::cerr << "[GBFS] error processing feed " << pf.id_ << " (" << pf.url_
                << "): " << ex.what() << "\n";

      // keep previous data
      if (prev_provider != nullptr) {
        provider.sys_info_ = prev_provider->sys_info_;
        provider.vehicle_types_ = prev_provider->vehicle_types_;
        provider.vehicle_types_map_ = prev_provider->vehicle_types_map_;
        provider.temp_vehicle_types_ = prev_provider->temp_vehicle_types_;
        provider.stations_ = prev_provider->stations_;
        provider.vehicle_status_ = prev_provider->vehicle_status_;
        provider.geofencing_zones_ = prev_provider->geofencing_zones_;
        provider.has_vehicles_to_rent_ = prev_provider->has_vehicles_to_rent_;
      }
    }

    if (data_changed) {
      try {
        partition_provider(provider);
        provider.has_vehicles_to_rent_ = utl::any_of(
            provider.products_,
            [](auto const& prod) { return prod.has_vehicles_to_rent_; });

        update_rtree(provider, prev_provider);

        d_->cache_.try_add_or_update(provider.idx_, [&]() {
          return compute_provider_routing_data(w_, l_, provider);
        });
      } catch (std::exception const& ex) {
        std::cerr << "[GBFS] error updating provider " << pf.id_ << ": "
                  << ex.what() << "\n";
      }
    } else if (prev_provider != nullptr) {
      // data not changed, copy previously computed products
      provider.products_ = prev_provider->products_;
      provider.has_vehicles_to_rent_ = prev_provider->has_vehicles_to_rent_;
    }
  }

  void partition_provider(gbfs_provider& provider) {
    if (provider.vehicle_types_.empty()) {
      auto& prod = provider.products_.emplace_back();
      prod.idx_ = gbfs_products_idx_t{0};
      prod.has_vehicles_to_rent_ =
          utl::any_of(provider.stations_,
                      [](auto const& st) {
                        return st.second.status_.is_renting_ &&
                               st.second.status_.num_vehicles_available_ > 0;
                      }) ||
          utl::any_of(provider.vehicle_status_, [](auto const& vs) {
            return !vs.is_disabled_ && !vs.is_reserved_;
          });
    } else {
      auto part = partition{vehicle_type_idx_t{provider.vehicle_types_.size()}};

      // refine by form factor + propulsion type
      auto by_form_factor =
          hash_map<std::pair<vehicle_form_factor, propulsion_type>,
                   std::vector<vehicle_type_idx_t>>{};
      for (auto const& vt : provider.vehicle_types_) {
        by_form_factor[std::pair{vt.form_factor_, vt.propulsion_type_}]
            .push_back(vt.idx_);
      }
      for (auto const& [_, vt_indices] : by_form_factor) {
        part.refine(vt_indices);
      }

      // refine by return constraints
      auto by_return_constraint =
          hash_map<return_constraint, std::vector<vehicle_type_idx_t>>{};
      for (auto const& vt : provider.vehicle_types_) {
        by_return_constraint[vt.return_constraint_].push_back(vt.idx_);
      }
      for (auto const& [_, vt_indices] : by_return_constraint) {
        part.refine(vt_indices);
      }

      // refine by known vs. guessed return constraints
      auto known_return_constraints = std::vector<vehicle_type_idx_t>{};
      auto guessed_return_constraints = std::vector<vehicle_type_idx_t>{};
      for (auto const& vt : provider.vehicle_types_) {
        if (vt.known_return_constraint_) {
          known_return_constraints.push_back(vt.idx_);
        } else {
          guessed_return_constraints.push_back(vt.idx_);
        }
      }
      part.refine(known_return_constraints);
      part.refine(guessed_return_constraints);

      // refine by return stations
      // TODO: only do this if the station is not in a zone where vehicles
      //   can be returned anywhere
      auto vts = std::vector<vehicle_type_idx_t>{};
      for (auto const& [id, st] : provider.stations_) {
        if (!st.status_.vehicle_docks_available_.empty()) {
          vts.clear();
          for (auto const& [vt, num] : st.status_.vehicle_docks_available_) {
            vts.push_back(vt);
          }
          part.refine(vts);
        }
      }

      // refine by geofencing zones
      for (auto const& z : provider.geofencing_zones_.zones_) {
        for (auto const& r : z.rules_) {
          part.refine(r.vehicle_type_idxs_);
        }
      }

      for (auto const& set : part.get_sets()) {
        auto const prod_idx = gbfs_products_idx_t{provider.products_.size()};
        auto& prod = provider.products_.emplace_back();
        prod.idx_ = prod_idx;
        prod.vehicle_types_ = set;
        auto const first_vt =
            provider.vehicle_types_.at(prod.vehicle_types_.front());
        prod.form_factor_ = first_vt.form_factor_;
        prod.propulsion_type_ = first_vt.propulsion_type_;
        prod.return_constraint_ = first_vt.return_constraint_;
        prod.known_return_constraint_ = first_vt.known_return_constraint_;
        prod.has_vehicles_to_rent_ =
            utl::any_of(provider.stations_,
                        [&](auto const& st) {
                          return st.second.status_.is_renting_ &&
                                 st.second.status_.num_vehicles_available_ > 0;
                        }) ||
            utl::any_of(provider.vehicle_status_, [&](auto const& vs) {
              return !vs.is_disabled_ && !vs.is_reserved_ &&
                     prod.includes_vehicle_type(vs.vehicle_type_idx_);
            });
      }
    }
  }

  void update_rtree(gbfs_provider const& provider,
                    gbfs_provider const* prev_provider) {
    auto added_stations = 0U;
    auto added_vehicles = 0U;
    auto removed_stations = 0U;
    auto removed_vehicles = 0U;
    auto moved_stations = 0U;
    auto moved_vehicles = 0U;

    if (prev_provider != nullptr) {
      using ST = std::pair<std::string, station>;
      utl::sorted_diff(
          prev_provider->stations_, provider.stations_,
          [](ST const& a, ST const& b) { return a.first < b.first; },
          [](ST const& a, ST const& b) {
            return a.second.info_.pos_ == b.second.info_.pos_;
          },
          utl::overloaded{
              [&](utl::op const o, ST const& s) {
                if (o == utl::op::kAdd) {
                  d_->provider_rtree_.add(s.second.info_.pos_, provider.idx_);
                  ++added_stations;
                } else {  // del
                  d_->provider_rtree_.remove(s.second.info_.pos_,
                                             provider.idx_);
                  ++removed_stations;
                }
              },
              [&](ST const& a, ST const& b) {
                d_->provider_rtree_.remove(a.second.info_.pos_, provider.idx_);
                d_->provider_rtree_.add(b.second.info_.pos_, provider.idx_);
                ++moved_stations;
              }});
      utl::sorted_diff(
          prev_provider->vehicle_status_, provider.vehicle_status_,
          [](vehicle_status const& a, vehicle_status const& b) {
            return a.id_ < b.id_;
          },
          [](vehicle_status const& a, vehicle_status const& b) {
            return a.pos_ == b.pos_;
          },
          utl::overloaded{
              [&](utl::op const o, vehicle_status const& v) {
                if (o == utl::op::kAdd) {
                  d_->provider_rtree_.add(v.pos_, provider.idx_);
                  ++added_vehicles;
                } else {  // del
                  d_->provider_rtree_.remove(v.pos_, provider.idx_);
                  ++removed_vehicles;
                }
              },
              [&](vehicle_status const& a, vehicle_status const& b) {
                d_->provider_rtree_.remove(a.pos_, provider.idx_);
                d_->provider_rtree_.add(b.pos_, provider.idx_);
                ++moved_vehicles;
              }});
    } else {
      for (auto const& station : provider.stations_) {
        d_->provider_rtree_.add(station.second.info_.pos_, provider.idx_);
        ++added_stations;
      }
      for (auto const& vehicle : provider.vehicle_status_) {
        if (vehicle.station_id_.empty()) {
          d_->provider_rtree_.add(vehicle.pos_, provider.idx_);
          ++added_vehicles;
        }
      }
    }
  }

  awaitable<void> init_aggregated_feed(std::string const& prefix,
                                       std::string const& url,
                                       headers_t const& headers,
                                       boost::json::object const& root) {
    auto af =
        d_->aggregated_feeds_
            ->emplace_back(std::make_unique<aggregated_feed>(aggregated_feed{
                .id_ = prefix,
                .url_ = url,
                .headers_ = headers,
                .expiry_ = get_expiry(root, std::chrono::hours{1})}))
            .get();

    co_return co_await process_aggregated_feed(*af, root);
  }

  awaitable<void> update_aggregated_feed(aggregated_feed& af) {
    if (af.needs_update()) {
      auto const file = co_await fetch_file("manifest", af.url_, af.headers_);
      co_await process_aggregated_feed(af, file.json_.as_object());
    } else {
      co_await update_aggregated_feed_provider_feeds(af);
    }
  }

  awaitable<void> process_aggregated_feed(aggregated_feed& af,
                                          boost::json::object const& root) {
    auto feeds = std::vector<provider_feed>{};
    if (root.contains("data") &&
        root.at("data").as_object().contains("datasets")) {
      // GBFS 3.x manifest.json
      for (auto const& dataset : root.at("data").at("datasets").as_array()) {
        auto const system_id =
            static_cast<std::string>(dataset.at("system_id").as_string());
        auto const combined_id = fmt::format("{}:{}", af.id_, system_id);

        auto const& versions = dataset.at("versions").as_array();
        if (versions.empty()) {
          continue;
        }
        // versions array must be sorted by increasing version number
        auto const& latest_version = versions.back().as_object();
        feeds.emplace_back(provider_feed{
            .id_ = combined_id,
            .url_ =
                static_cast<std::string>(latest_version.at("url").as_string()),
            .headers_ = af.headers_,
            .default_restrictions_ =
                lookup_default_restrictions(af.id_, combined_id),
            .default_return_constraint_ =
                lookup_default_return_constraint(af.id_, combined_id)});
      }
    } else if (root.contains("systems")) {
      // Lamassu 2.3 format
      for (auto const& system : root.at("systems").as_array()) {
        auto const system_id =
            static_cast<std::string>(system.at("id").as_string());
        auto const combined_id = fmt::format("{}:{}", af.id_, system_id);
        feeds.emplace_back(provider_feed{
            .id_ = combined_id,
            .url_ = static_cast<std::string>(system.at("url").as_string()),
            .headers_ = af.headers_,
            .default_restrictions_ =
                lookup_default_restrictions(af.id_, combined_id),
            .default_return_constraint_ =
                lookup_default_return_constraint(af.id_, combined_id)});
      }
    }

    af.feeds_ = std::move(feeds);
    co_await update_aggregated_feed_provider_feeds(af);
  }

  awaitable<void> update_aggregated_feed_provider_feeds(aggregated_feed& af) {
    auto executor = co_await asio::this_coro::executor;
    co_await asio::experimental::make_parallel_group(
        utl::to_vec(af.feeds_,
                    [&](auto const& pf) {
                      return boost::asio::co_spawn(
                          executor,
                          [this, pf = &pf]() -> awaitable<void> {
                            co_await update_provider_feed(*pf);
                          },
                          asio::deferred);
                    }))
        .async_wait(asio::experimental::wait_for_all(), asio::use_awaitable);
  }

  awaitable<gbfs_file> fetch_file(
      std::string_view const name,
      std::string_view const url,
      headers_t const& headers,
      std::optional<std::filesystem::path> const& dir = std::nullopt) {
    auto content = std::string{};
    if (dir.has_value()) {
      content = read_file(*dir / fmt::format("{}.json", name));
    } else {
      auto const res = co_await client_.get(boost::urls::url{url}, headers);
      content = get_http_body(res);
    }
    auto j = json::parse(content);
    auto j_root = j.as_object();
    auto const next_refresh =
        std::chrono::system_clock::now() +
        std::chrono::seconds{
            j_root.contains("ttl") ? j_root.at("ttl").to_number<int>() : 0};
    co_return gbfs_file{.json_ = std::move(j),
                        .hash_ = hash_gbfs_data(content),
                        .next_refresh_ = next_refresh};
  }

  geofencing_restrictions lookup_default_restrictions(std::string const& prefix,
                                                      std::string const& id) {
    auto const convert = [&](config::gbfs::restrictions const& r) {
      return geofencing_restrictions{
          .ride_start_allowed_ = r.ride_start_allowed_,
          .ride_end_allowed_ = r.ride_end_allowed_,
          .ride_through_allowed_ = r.ride_through_allowed_,
          .station_parking_ = r.station_parking_};
    };

    if (auto const it = c_.default_restrictions_.find(id);
        it != end(c_.default_restrictions_)) {
      return convert(it->second);
    } else if (auto const prefix_it = c_.default_restrictions_.find(prefix);
               prefix_it != end(c_.default_restrictions_)) {
      return convert(prefix_it->second);
    } else {
      return {};
    }
  }

  std::optional<return_constraint> lookup_default_return_constraint(
      std::string const& prefix, std::string const& id) {
    auto const convert = [&](config::gbfs::restrictions const& r) {
      return r.return_constraint_.has_value()
                 ? parse_return_constraint(r.return_constraint_.value())
                 : std::nullopt;
    };
    if (auto const it = c_.default_restrictions_.find(id);
        it != end(c_.default_restrictions_)) {
      return convert(it->second);
    } else if (auto const prefix_it = c_.default_restrictions_.find(prefix);
               prefix_it != end(c_.default_restrictions_)) {
      return convert(prefix_it->second);
    } else {
      return {};
    }
  }

  config::gbfs const& c_;
  osr::ways const& w_;
  osr::lookup const& l_;

  gbfs_data* d_;
  gbfs_data const* prev_d_;

  http_client client_;
};

awaitable<void> update(config const& c,
                       osr::ways const& w,
                       osr::lookup const& l,
                       std::shared_ptr<gbfs_data>& data_ptr) {
  auto const t = utl::scoped_timer{"gbfs::update"};

  if (!c.gbfs_.has_value()) {
    co_return;
  }

  auto const prev_d = data_ptr;
  auto const d = std::make_shared<gbfs_data>(c.gbfs_->cache_size_);

  auto update = gbfs_update{*c.gbfs_, w, l, d.get(), prev_d.get()};
  co_await update.run();
  data_ptr = d;
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
