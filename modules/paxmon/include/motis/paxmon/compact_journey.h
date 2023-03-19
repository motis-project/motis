#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "cista/hashing.h"

#include "utl/verify.h"
#include "utl/zip.h"

#include "motis/vector.h"

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

struct final_footpath {
  CISTA_COMPARABLE()

  inline bool is_footpath() const {
    return from_station_id_ != 0 && to_station_id_ != 0;
  }

  cista::hash_t hash() const {
    return cista::build_hash(duration_, from_station_id_, to_station_id_);
  }

  duration duration_{0};
  unsigned from_station_id_{0};
  unsigned to_station_id_{0};
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
    auto const& this_ffp = static_cast<Derived const&>(*this).final_footpath();
    auto const& other_ffp =
        static_cast<OtherDerived const&>(o).final_footpath();
    return this_ffp == other_ffp;
  }

  template <typename OtherDerived>
  inline bool operator!=(compact_journey_base<OtherDerived> const& o) const {
    return !(*this == o);
  }

  inline unsigned start_station_id() const {
    return static_cast<Derived const&>(*this).legs().front().enter_station_id_;
  }

  inline unsigned destination_station_id() const {
    auto const& ffp = static_cast<Derived const&>(*this).final_footpath();
    if (ffp.is_footpath()) {
      return ffp.to_station_id_;
    } else {
      return static_cast<Derived const&>(*this).legs().back().exit_station_id_;
    }
  }

  inline duration scheduled_duration() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    auto const& ffp = static_cast<Derived const&>(*this).final_footpath();
    return (!legs.empty() ? legs.back().exit_time_ - legs.front().enter_time_
                          : 0) +
           ffp.duration_;
  }

  inline time scheduled_departure_time() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    return !legs.empty() ? legs.front().enter_time_ : INVALID_TIME;
  }

  inline time scheduled_arrival_time() const {
    auto const& legs = static_cast<Derived const&>(*this).legs();
    auto const& ffp = static_cast<Derived const&>(*this).final_footpath();
    return !legs.empty() ? legs.back().exit_time_ + ffp.duration_
                         : INVALID_TIME;
  }

  cista::hash_t hash() const {
    auto h = cista::BASE_HASH;
    for (auto const& leg : static_cast<Derived const&>(*this).legs()) {
      h = cista::hash_combine(h, leg.hash());
    }
    h = cista::hash_combine(
        h, static_cast<Derived const&>(*this).final_footpath().hash());
    return h;
  }
};

struct compact_journey : public compact_journey_base<compact_journey> {
  compact_journey() = default;
  compact_journey(std::vector<journey_leg>&& legs, final_footpath const& ffp)
      : legs_{std::move(legs)}, final_footpath_{ffp} {}

  inline std::vector<journey_leg>& legs() { return legs_; }
  inline std::vector<journey_leg> const& legs() const { return legs_; }

  inline struct final_footpath& final_footpath() { return final_footpath_; }
  inline struct final_footpath const& final_footpath() const {
    return final_footpath_;
  }

  auto cista_members() noexcept { return std::tie(legs_, final_footpath_); }

  std::vector<journey_leg> legs_;
  struct final_footpath final_footpath_ {};
};

struct fws_compact_journey : public compact_journey_base<fws_compact_journey> {
  using fws_type = dynamic_fws_multimap<journey_leg>;
  using bucket_type = typename fws_type::const_bucket;
  using index_type = typename fws_type::size_type;

  fws_compact_journey(bucket_type const& bucket,
                      mcd::vector<final_footpath> const& ffp_vector)
      : bucket_{bucket}, ffp_vector_{ffp_vector} {}

  inline bucket_type const& legs() const { return bucket_; }
  inline index_type index() const { return bucket_.index(); }
  inline struct final_footpath const& final_footpath() const {
    return ffp_vector_.at(index());
  }

  auto cista_members() noexcept { return std::tie(bucket_, ffp_vector_); }

  bucket_type const bucket_;
  mcd::vector<struct final_footpath> const& ffp_vector_;
};

inline compact_journey to_compact_journey(compact_journey const& cj) {
  return cj;
}

inline compact_journey to_compact_journey(fws_compact_journey const& cj) {
  return compact_journey{
      std::vector<journey_leg>(cj.legs().begin(), cj.legs().end()),
      cj.final_footpath()};
}

inline fws_compact_journey to_fws_compact_journey(
    typename fws_compact_journey::fws_type& fws,
    mcd::vector<final_footpath>& ffp_vector, compact_journey const& cj) {
  auto bucket = fws.emplace_back();
  bucket.reserve(cj.legs().size());
  for (auto const& leg : cj.legs()) {
    bucket.push_back(leg);
  }
  auto const index = bucket.index();
  ffp_vector.emplace_back(cj.final_footpath());
  utl::verify(ffp_vector.size() == index + 1,
              "to_fws_compact_journey: vector size mismatch");
  return fws_compact_journey{bucket, ffp_vector};
}

}  // namespace motis::paxmon
