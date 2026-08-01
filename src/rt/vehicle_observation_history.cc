#include "motis/rt/vehicle_observation_history.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>

namespace motis {
namespace {

auto order_key(vehicle_observation const& observation) {
  return std::pair{observation_time(observation), observation.ingested_time_};
}

bool same_observation(vehicle_observation const& a,
                      vehicle_observation const& b) {
  // A reported timestamp identifies the provider observation across repeated
  // feed deliveries. Without one, ingest time is the only safe time identity.
  auto const same_time =
      a.reported_time_.has_value() && b.reported_time_.has_value()
          ? a.reported_time_ == b.reported_time_
          : a.reported_time_ == b.reported_time_ &&
                a.ingested_time_ == b.ingested_time_;
  return same_time && a.feed_id_ == b.feed_id_ &&
         a.entity_id_ == b.entity_id_ && a.vehicle_id_ == b.vehicle_id_ &&
         a.trip_ == b.trip_ && a.latitude_ == b.latitude_ &&
         a.longitude_ == b.longitude_ && a.bearing_ == b.bearing_ &&
         a.speed_mps_ == b.speed_mps_ &&
         a.current_stop_sequence_ == b.current_stop_sequence_ &&
         a.stop_id_ == b.stop_id_;
}

void hash_combine(std::size_t& seed, std::size_t const value) {
  seed ^= value + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
}

}  // namespace

std::optional<vehicle_key> make_vehicle_key(
    vehicle_observation const& observation) {
  if (observation.vehicle_id_.has_value() &&
      !observation.vehicle_id_->empty()) {
    return vehicle_key{observation.feed_id_, *observation.vehicle_id_,
                       vehicle_key_source::kVehicleDescriptor};
  }
  if (!observation.entity_id_.empty()) {
    return vehicle_key{observation.feed_id_, observation.entity_id_,
                       vehicle_key_source::kEntityId};
  }
  return std::nullopt;
}

std::int64_t observation_time(vehicle_observation const& observation) {
  return observation.reported_time_.value_or(observation.ingested_time_);
}

std::size_t vehicle_observation_history::vehicle_key_hash::operator()(
    vehicle_key const& key) const noexcept {
  auto seed = std::hash<std::string>{}(key.feed_id_);
  hash_combine(seed, std::hash<std::string>{}(key.stable_id_));
  hash_combine(seed, std::hash<unsigned>{}(static_cast<unsigned>(key.source_)));
  return seed;
}

std::size_t vehicle_observation_history::entity_key_hash::operator()(
    entity_key const& key) const noexcept {
  auto seed = std::hash<std::string>{}(key.feed_id_);
  hash_combine(seed, std::hash<std::string>{}(key.entity_id_));
  return seed;
}

bool vehicle_observation_history::ingest(
    vehicle_observation observation, observation_history_policy const& policy) {
  auto const now = observation.ingested_time_;
  auto const accepted = ingest_unpruned(std::move(observation));
  prune(now, policy);
  return accepted;
}

bool vehicle_observation_history::ingest_unpruned(
    vehicle_observation observation,
    std::span<batch_locator const> batch_locators,
    locator_evidence const evidence) {
  auto const key = make_vehicle_key(observation);
  auto const disposition =
      classify_unpruned(observation, batch_locators, evidence);
  if (disposition == ingest_disposition::kInvalid) {
    return false;
  }
  if (disposition == ingest_disposition::kRejected) {
    return true;
  }

  auto const entity = entity_key{observation.feed_id_, observation.entity_id_};
  if (!observation.entity_id_.empty()) {
    if (auto const old = key_by_entity_.find(entity);
        old != end(key_by_entity_) && old->second != *key) {
      // The same feed entity changed from descriptor identity to fallback (or
      // vice versa), or now names a different stable vehicle. Mixing those
      // histories would make trip continuity unsafe. Reassigning just this
      // locator is safe when the old vehicle is represented elsewhere;
      // otherwise only a genuinely newer observation may replace a vehicle
      // that is still current. Retained history must not reserve an inactive
      // entity locator.
      if (!represented_elsewhere(old->second, entity, observation.entity_id_,
                                 batch_locators, evidence)) {
        erase_history(old->second);
      }
    }
    key_by_entity_.insert_or_assign(entity, *key);
  }

  auto const [history_it, inserted] =
      histories_.try_emplace(*key, history_entry{.trip_ = observation.trip_});
  auto& history = history_it->second;
  if (!inserted && history.trip_ != observation.trip_) {
    // Retaining observations from different trip instances in one history is
    // unsafe, but an out-of-order observation from a prior trip is not a valid
    // reason to discard newer progress.
    history.trip_ = observation.trip_;
    history.observations_.clear();
    current_.erase(*key);
  }

  auto const duplicate = std::ranges::find_if(
      history.observations_, [&](vehicle_observation const& existing) {
        return same_observation(existing, observation);
      });
  if (duplicate != end(history.observations_)) {
    duplicate->ingested_time_ =
        std::max(duplicate->ingested_time_, observation.ingested_time_);
  } else {
    history.observations_.emplace_back(observation);
  }
  std::ranges::sort(history.observations_, {}, order_key);

  auto const current = current_.find(*key);
  if (current == end(current_) ||
      order_key(current->second) <= order_key(observation)) {
    current_.insert_or_assign(*key, std::move(observation));
  }
  return true;
}

vehicle_observation_history::ingest_disposition
vehicle_observation_history::classify_unpruned(
    vehicle_observation const& observation,
    std::span<batch_locator const> batch_locators,
    locator_evidence const evidence) const {
  auto const key = make_vehicle_key(observation);
  if (!key.has_value()) {
    return ingest_disposition::kInvalid;
  }

  auto const entity = entity_key{observation.feed_id_, observation.entity_id_};
  if (!observation.entity_id_.empty()) {
    if (auto const old = key_by_entity_.find(entity);
        old != end(key_by_entity_) && old->second != *key) {
      auto const old_current = current_.find(old->second);
      if (!represented_elsewhere(old->second, entity, observation.entity_id_,
                                 batch_locators, evidence) &&
          old_current != end(current_) &&
          !is_strictly_newer_than_history(old->second, observation)) {
        return ingest_disposition::kRejected;
      }
    }
  }

  if (auto const history = histories_.find(*key);
      history != end(histories_) &&
      history->second.trip_ != observation.trip_ &&
      !is_strictly_newer_than_history(*key, observation)) {
    return ingest_disposition::kRejected;
  }
  return ingest_disposition::kAccepted;
}

bool vehicle_observation_history::represented_elsewhere(
    vehicle_key const& key,
    entity_key const& entity,
    std::string_view const entity_id,
    std::span<batch_locator const> batch_locators,
    locator_evidence const evidence) const {
  if (std::ranges::any_of(batch_locators, [&](auto const& locator) {
        return locator.entity_id_ != entity_id && locator.key_ == key;
      })) {
    return true;
  }
  if (evidence != locator_evidence::kBatchOrCurrent) {
    return false;
  }
  auto const old_current = current_.find(key);
  return std::ranges::any_of(key_by_entity_, [&](auto const& item) {
    if (item.first == entity || item.second != key) {
      return false;
    }
    return old_current != end(current_) &&
           old_current->second.entity_id_ == item.first.entity_id_;
  });
}

void vehicle_observation_history::replace_feed(
    std::string_view const feed_id,
    std::span<vehicle_observation const> observations,
    std::int64_t const now,
    observation_history_policy const& policy) {
  auto batch_locators = std::vector<batch_locator>{};
  auto dispositions = std::vector<ingest_disposition>{};
  dispositions.reserve(observations.size());
  for (auto observation : observations) {
    observation.feed_id_ = feed_id;
    auto const disposition =
        classify_unpruned(observation, {}, locator_evidence::kBatchOnly);
    dispositions.emplace_back(disposition);
    if (auto const key = make_vehicle_key(observation);
        disposition == ingest_disposition::kAccepted && key.has_value() &&
        !observation.entity_id_.empty()) {
      batch_locators.emplace_back(*key, observation.entity_id_);
    }
  }

  auto absent = std::unordered_set<vehicle_key, vehicle_key_hash>{};
  for (auto const& [key, _] : current_) {
    if (key.feed_id_ == feed_id) {
      absent.emplace(key);
    }
  }

  for (auto idx = 0U; idx != observations.size(); ++idx) {
    auto observation = observations[idx];
    observation.feed_id_ = feed_id;
    auto represented = std::vector<vehicle_key>{};
    if (auto const key = make_vehicle_key(observation); key.has_value()) {
      represented.emplace_back(*key);
    }
    if (!observation.entity_id_.empty()) {
      auto const locator =
          entity_key{observation.feed_id_, observation.entity_id_};
      if (auto const previous = key_by_entity_.find(locator);
          previous != end(key_by_entity_)) {
        represented.emplace_back(previous->second);
      }
    }

    auto const ingested =
        dispositions[idx] == ingest_disposition::kRejected ||
        (dispositions[idx] == ingest_disposition::kAccepted &&
         ingest_unpruned(std::move(observation), batch_locators,
                         locator_evidence::kBatchOnly));
    if (ingested) {
      for (auto const& key : represented) {
        absent.erase(key);
      }
    }
  }
  for (auto const& key : absent) {
    current_.erase(key);
  }
  prune(now, policy);
}

void vehicle_observation_history::update_feed(
    std::string_view const feed_id,
    std::span<vehicle_observation const> observations,
    std::span<std::string const> deleted_entity_ids,
    std::int64_t const now,
    observation_history_policy const& policy) {
  erase_deleted(feed_id, deleted_entity_ids);
  ingest_feed(feed_id, observations);
  prune(now, policy);
}

void vehicle_observation_history::ingest_feed(
    std::string_view const feed_id,
    std::span<vehicle_observation const> observations) {
  auto batch_locators = std::vector<batch_locator>{};
  for (auto observation : observations) {
    observation.feed_id_ = feed_id;
    if (auto const key = make_vehicle_key(observation);
        key.has_value() && !observation.entity_id_.empty()) {
      batch_locators.emplace_back(*key, observation.entity_id_);
    }
  }
  for (auto observation : observations) {
    observation.feed_id_ = feed_id;
    ingest_unpruned(std::move(observation), batch_locators);
  }
}

void vehicle_observation_history::erase_deleted(
    std::string_view const feed_id, std::span<std::string const> entity_ids) {
  for (auto const& entity_id : entity_ids) {
    auto const locator = entity_key{std::string{feed_id}, entity_id};
    auto const key = key_by_entity_.find(locator);
    if (key == end(key_by_entity_)) {
      continue;
    }
    auto const current = current_.find(key->second);
    if (current != end(current_) && current->second.entity_id_ == entity_id) {
      current_.erase(current);
    }
  }
}

void vehicle_observation_history::prune(
    std::int64_t const now, observation_history_policy const& policy) {
  auto const cutoff = now - policy.max_age_.count();
  for (auto& [_, history] : histories_) {
    std::erase_if(history.observations_, [&](vehicle_observation const& x) {
      return observation_time(x) < cutoff;
    });
    if (history.observations_.size() > policy.max_observations_per_vehicle_) {
      history.observations_.erase(
          begin(history.observations_),
          end(history.observations_) -
              static_cast<std::ptrdiff_t>(
                  policy.max_observations_per_vehicle_));
    }
  }

  std::erase_if(current_, [&](auto const& item) {
    return observation_time(item.second) < cutoff;
  });

  auto removed = std::vector<vehicle_key>{};
  for (auto const& [key, history] : histories_) {
    if (history.observations_.empty()) {
      removed.emplace_back(key);
    }
  }
  for (auto const& key : removed) {
    erase_history(key);
  }

  // Entity IDs can rotate while a stable vehicle descriptor remains the same.
  // Keep locator state only while an observation carrying that entity is still
  // retained, so the auxiliary identity index is bounded with the histories.
  std::erase_if(key_by_entity_, [&](auto const& item) {
    auto const history = histories_.find(item.second);
    return history == end(histories_) ||
           std::ranges::none_of(history->second.observations_,
                                [&](vehicle_observation const& x) {
                                  return x.entity_id_ == item.first.entity_id_;
                                });
  });
}

std::span<vehicle_observation const> vehicle_observation_history::observations(
    vehicle_key const& key) const {
  auto const it = histories_.find(key);
  return it == end(histories_)
             ? std::span<vehicle_observation const>{}
             : std::span<vehicle_observation const>{it->second.observations_};
}

vehicle_observation const* vehicle_observation_history::effective_observation(
    vehicle_key const& key) const {
  auto const history = observations(key);
  return history.empty() ? nullptr : &history.back();
}

vehicle_observation const* vehicle_observation_history::current_observation(
    vehicle_key const& key) const {
  auto const it = current_.find(key);
  return it == end(current_) ? nullptr : &it->second;
}

std::size_t vehicle_observation_history::active_histories() const {
  return histories_.size();
}

std::size_t vehicle_observation_history::observation_count() const {
  auto count = std::size_t{};
  for (auto const& [_, history] : histories_) {
    count += history.observations_.size();
  }
  return count;
}

bool vehicle_observation_history::is_strictly_newer_than_history(
    vehicle_key const& key, vehicle_observation const& observation) const {
  auto const history = histories_.find(key);
  return history == end(histories_) || history->second.observations_.empty() ||
         order_key(history->second.observations_.back()) <
             order_key(observation);
}

void vehicle_observation_history::erase_history(vehicle_key const& key) {
  histories_.erase(key);
  current_.erase(key);
  std::erase_if(key_by_entity_,
                [&](auto const& item) { return item.second == key; });
}

}  // namespace motis
