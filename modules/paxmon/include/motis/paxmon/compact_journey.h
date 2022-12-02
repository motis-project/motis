#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "cista/hashing.h"

#include "utl/zip.h"

#include "motis/core/common/dynamic_fws_multimap.h"

#include "motis/core/schedule/time.h"
#include "motis/core/schedule/trip.h"

#include "motis/paxmon/transfer_info.h"

namespace motis::paxmon {

struct journey_leg {
  CISTA_COMPARABLE()

  cista::hash_t hash() const {
    return cista::build_hash(trip_idx_, enter_station_id_, exit_station_id_,
                             enter_time_, exit_time_);
  }

  trip_idx_t trip_idx_{0};
  unsigned enter_station_id_{0};
  unsigned exit_station_id_{0};
  motis::time enter_time_{0};
  motis::time exit_time_{0};
  std::optional<transfer_info> enter_transfer_;
};

template <typename Derived>
struct compact_journey_base {
  template <typename OtherDerived>
  inline bool operator==(compact_journey_base<OtherDerived> const& o) const {
    auto const& this_legs = static_cast<Derived const&>(*this).legs();
    auto const& other_legs = static_cast<OtherDerived const&>(o).legs();
    if (this_legs.size() != other_legs.size()) {
      return false;
    }
    for (auto const& [a, b] : utl::zip(this_legs, other_legs)) {
      if (a != b) {
        return false;
      }
    }
    return true;
  }

  template <typename OtherDerived>
  inline bool operator!=(compact_journey_base<OtherDerived> const& o) const {
    return !(*this == o);
  }

  inline unsigned start_station_id() const {
    return static_cast<Derived const&>(*this).legs().front().enter_station_id_;
  }

  inline unsigned destination_station_id() const {
    return static_cast<Derived const&>(*this).legs().back().exit_station_id_;
  }

  inline duration scheduled_duration() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    return !legs.empty() ? legs.back().exit_time_ - legs.front().enter_time_
                         : 0;
  }

  inline time scheduled_departure_time() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    return !legs.empty() ? legs.front().enter_time_ : INVALID_TIME;
  }

  inline time scheduled_arrival_time() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    return !legs.empty() ? legs.back().exit_time_ : INVALID_TIME;
  }

  cista::hash_t hash() const {
    auto h = cista::BASE_HASH;
    for (auto const& leg : static_cast<Derived const&>(*this).legs()) {
      h = cista::hash_combine(h, leg.hash());
    }
    return h;
  }
};

struct compact_journey : public compact_journey_base<compact_journey> {
  compact_journey() = default;
  explicit compact_journey(std::vector<journey_leg>&& legs)
      : legs_{std::move(legs)} {}

  inline std::vector<journey_leg>& legs() { return legs_; }
  inline std::vector<journey_leg> const& legs() const { return legs_; }

  auto cista_members() noexcept { return std::tie(legs_); }

  std::vector<journey_leg> legs_;
};

struct fws_compact_journey : public compact_journey_base<fws_compact_journey> {
  using fws_type = dynamic_fws_multimap<journey_leg>;
  using bucket_type = typename fws_type::const_bucket;
  using index_type = typename fws_type::size_type;

  explicit fws_compact_journey(bucket_type const& bucket) : bucket_{bucket} {}

  inline bucket_type const& legs() const { return bucket_; }
  inline index_type index() const { return bucket_.index(); }

  auto cista_members() noexcept { return std::tie(bucket_); }

  bucket_type const bucket_;
};

inline compact_journey to_compact_journey(compact_journey const& cj) {
  return cj;
}

inline compact_journey to_compact_journey(fws_compact_journey const& cj) {
  return compact_journey{
      std::vector<journey_leg>(cj.legs().begin(), cj.legs().end())};
}

inline fws_compact_journey to_fws_compact_journey(
    typename fws_compact_journey::fws_type& fws, compact_journey const& cj) {
  auto bucket = fws.emplace_back();
  bucket.reserve(cj.legs().size());
  for (auto const& leg : cj.legs()) {
    bucket.push_back(leg);
  }
  return fws_compact_journey{bucket};
}

}  // namespace motis::paxmon
