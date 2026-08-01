#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace motis {

struct observation_history_policy {
  std::chrono::seconds max_age_;
  std::size_t max_observations_per_vehicle_;
};

enum class vehicle_key_source { kVehicleDescriptor, kEntityId };

struct vehicle_key {
  std::string feed_id_;
  std::string stable_id_;
  vehicle_key_source source_{vehicle_key_source::kEntityId};

  bool operator==(vehicle_key const&) const = default;
};

struct vehicle_trip_instance {
  std::optional<std::string> trip_id_;
  std::optional<std::string> start_date_;
  std::optional<std::string> start_time_;

  bool operator==(vehicle_trip_instance const&) const = default;
};

struct vehicle_observation {
  std::string feed_id_;
  std::string entity_id_;
  std::optional<std::string> vehicle_id_;
  vehicle_trip_instance trip_;
  double latitude_{};
  double longitude_{};
  std::optional<double> bearing_;
  std::optional<double> speed_mps_;
  std::optional<std::uint32_t> current_stop_sequence_;
  std::optional<std::string> stop_id_;
  std::optional<std::int64_t> reported_time_;
  std::int64_t ingested_time_{};
};

[[nodiscard]] std::optional<vehicle_key> make_vehicle_key(
    vehicle_observation const&);

[[nodiscard]] std::int64_t observation_time(vehicle_observation const&);

struct vehicle_observation_history {
  // Ingests one observation and prunes relative to its ingest time. Returns
  // false when the observation has no stable vehicle or entity identity.
  bool ingest(vehicle_observation, observation_history_policy const&);

  // A full replacement invalidates absent current observations, but
  // intentionally retains their short histories until they expire by policy.
  void replace_feed(std::string_view,
                    std::span<vehicle_observation const>,
                    std::int64_t,
                    observation_history_policy const&);

  // A differential update changes only named observations. Deleted entities
  // cease to be current immediately while their histories remain available
  // until prune.
  void update_feed(std::string_view,
                   std::span<vehicle_observation const>,
                   std::span<std::string const>,
                   std::int64_t,
                   observation_history_policy const&);

  void erase_deleted(std::string_view, std::span<std::string const>);

  void prune(std::int64_t, observation_history_policy const&);

  [[nodiscard]] std::span<vehicle_observation const> observations(
      vehicle_key const&) const;

  // The effective observation is the newest by reported time (falling back to
  // ingest time), then ingest time. Late arrivals can therefore be retained
  // without moving a progress consumer back to an older observation.
  [[nodiscard]] vehicle_observation const* effective_observation(
      vehicle_key const&) const;

  [[nodiscard]] vehicle_observation const* current_observation(
      vehicle_key const&) const;

  [[nodiscard]] std::size_t active_histories() const;
  [[nodiscard]] std::size_t observation_count() const;

private:
  struct vehicle_key_hash {
    std::size_t operator()(vehicle_key const&) const noexcept;
  };

  struct entity_key {
    std::string feed_id_;
    std::string entity_id_;

    bool operator==(entity_key const&) const = default;
  };

  struct entity_key_hash {
    std::size_t operator()(entity_key const&) const noexcept;
  };

  struct history_entry {
    vehicle_trip_instance trip_;
    std::vector<vehicle_observation> observations_;
  };

  struct batch_locator {
    vehicle_key key_;
    std::string entity_id_;
  };

  enum class locator_evidence { kBatchOnly, kBatchOrCurrent };
  enum class ingest_disposition { kInvalid, kRejected, kAccepted };

  bool ingest_unpruned(vehicle_observation,
                       std::span<batch_locator const> batch_locators = {},
                       locator_evidence = locator_evidence::kBatchOrCurrent);
  ingest_disposition classify_unpruned(
      vehicle_observation const&,
      std::span<batch_locator const> batch_locators = {},
      locator_evidence = locator_evidence::kBatchOrCurrent) const;
  bool represented_elsewhere(vehicle_key const&,
                             entity_key const&,
                             std::string_view,
                             std::span<batch_locator const>,
                             locator_evidence) const;
  void ingest_feed(std::string_view, std::span<vehicle_observation const>);
  bool is_strictly_newer_than_history(vehicle_key const&,
                                      vehicle_observation const&) const;
  void erase_history(vehicle_key const&);

  std::unordered_map<vehicle_key, history_entry, vehicle_key_hash> histories_;
  std::unordered_map<vehicle_key, vehicle_observation, vehicle_key_hash>
      current_;
  std::unordered_map<entity_key, vehicle_key, entity_key_hash> key_by_entity_;
};

}  // namespace motis
