#pragma once

#include <cinttypes>
#include <optional>
#include <utility>

#include "cista/hashing.h"
#include "cista/offset_t.h"
#include "cista/reflection/comparable.h"

#include "utl/pipes.h"
#include "utl/to_vec.h"

#include "motis/string.h"
#include "motis/vector.h"

#include "motis/core/common/hash_helper.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/edges.h"
#include "motis/core/schedule/nodes.h"

namespace motis {

struct primary_trip_id {
  primary_trip_id()
      : station_id_{0}, first_departure_mam_{INVALID_MAM}, train_nr_{0} {}
  primary_trip_id(uint32_t station_id, uint32_t train_nr,
                  mam_t first_departure_mam)
      : station_id_(station_id),
        first_departure_mam_(first_departure_mam),
        train_nr_(train_nr) {}

  friend bool operator<(primary_trip_id const& lhs,
                        primary_trip_id const& rhs) {
    uint64_t a = 0, b = 0;
    std::memcpy(&a, &lhs, sizeof(a));
    std::memcpy(&b, &rhs, sizeof(b));
    return a < b;
  }

  friend bool operator==(primary_trip_id const& lhs,
                         primary_trip_id const& rhs) {
    uint64_t a = 0, b = 0;
    std::memcpy(&a, &lhs, sizeof(a));
    std::memcpy(&b, &rhs, sizeof(b));
    return a == b;
  }

  friend std::ostream& operator<<(std::ostream& out,
                                  primary_trip_id const& id) {
    return out << "{PRIMARY_TRIP_ID station_idx=" << id.station_id_
               << ", first_dep="
               << format_time(time{day_idx_t{0}, id.first_departure_mam()})
               << ", train_nr=" << id.train_nr_ << "}";
  }

  uint32_t station_id() const { return static_cast<uint32_t>(station_id_); }
  uint32_t train_nr() const { return static_cast<uint32_t>(train_nr_); }
  mam_t first_departure_mam() const {
    return static_cast<mam_t>(first_departure_mam_);
  }

  uint64_t station_id_ : 31;
  uint64_t first_departure_mam_ : 16;
  uint64_t train_nr_ : 17;
};

static_assert(sizeof(primary_trip_id) == 8);

struct secondary_trip_id {
  CISTA_COMPARABLE();
  uint32_t target_station_id_{0U};
  mam_t last_arrival_mam_{INVALID_MAM};
  mcd::string line_id_;
};

struct trip_debug {
  friend std::ostream& operator<<(std::ostream& out, trip_debug const& dbg) {
    return out << dbg.str();
  }

  std::string str() const {
    return file_ == nullptr ? ""
                            : static_cast<std::string>(*file_) + ":" +
                                  std::to_string(line_from_) + ":" +
                                  std::to_string(line_to_);
  }

  mcd::string* file_{nullptr};
  int line_from_{0}, line_to_{0};
};

struct full_trip_id {
  CISTA_COMPARABLE()
  primary_trip_id primary_;
  secondary_trip_id secondary_;
};

struct gtfs_trip_id {
  gtfs_trip_id() = default;
  gtfs_trip_id(std::string const& dataset_prefix, std::string const& trip_id,
               std::optional<unixtime> start_date)
      : trip_id_{dataset_prefix + trip_id}, start_date_{start_date} {}
  friend std::ostream& operator<<(std::ostream& out, gtfs_trip_id const&);
  bool operator<(gtfs_trip_id const& o) const {
    return std::tie(trip_id_, start_date_) <
           std::tie(o.trip_id_, o.start_date_);
  }
  bool operator==(gtfs_trip_id const& o) const {
    return std::tie(trip_id_, start_date_) ==
           std::tie(o.trip_id_, o.start_date_);
  }
  bool operator!=(gtfs_trip_id const& o) const {
    return std::tie(trip_id_, start_date_) !=
           std::tie(o.trip_id_, o.start_date_);
  }
  mcd::string trip_id_;
  std::optional<unixtime> start_date_;
};

struct trip_info;

struct concrete_trip {
  CISTA_COMPARABLE()

  time get_first_dep_time() const;
  time get_last_arr_time() const;
  generic_light_connection lcon(size_t) const;

  trip_info const* trp_{nullptr};
  day_idx_t day_idx_{INVALID_TIME.day()};
};

struct trip_info {
  struct route_edge {
    route_edge() = default;

    route_edge(edge const* e) {  // NOLINT
      if (e != nullptr) {
        route_node_ = e->from();
        for (auto i = 0U; i < route_node_->edges_.size(); ++i) {
          if (&route_node_->edges_[i] == e) {
            outgoing_edge_idx_ = i;
            return;
          }
        }
        assert(false);
      }
    }

#if defined(MOTIS_SCHEDULE_MODE_OFFSET)
    route_edge(ptr<edge const> e) : route_edge{e.get()} {}  // NOLINT
#endif

    friend bool operator==(route_edge const& a, route_edge const& b) {
      return std::tie(a.route_node_, a.outgoing_edge_idx_) ==
             std::tie(b.route_node_, b.outgoing_edge_idx_);
    }

    friend bool operator<(route_edge const& a, route_edge const& b) {
      return std::tie(a.route_node_, a.outgoing_edge_idx_) <
             std::tie(b.route_node_, b.outgoing_edge_idx_);
    }

    bool is_not_null() const { return route_node_ != nullptr; }

    edge* get_edge() const {
      assert(outgoing_edge_idx_ < route_node_->edges_.size());
      return &route_node_->edges_[outgoing_edge_idx_];
    }

    edge* operator->() const { return get_edge(); }

    operator edge*() const { return get_edge(); }  // NOLINT

    cista::hash_t hash() const {
      return cista::build_hash(route_node_ != nullptr ? route_node_->id_ : 0U,
                               outgoing_edge_idx_);
    }

    ptr<node> route_node_{nullptr};
    uint32_t outgoing_edge_idx_{0};
  };

  auto concrete_trips() const {
    return utl::iota(day_idx_t{0}, MAX_DAYS)  //
           | utl::remove_if([&](auto const day) {
               if (edges_->front()->empty()) {
                 return true;
               }

               return !edges_->front()
                           ->static_lcons()
                           .at(lcon_idx_)
                           .traffic_days_->test(day);
             })  //
           | utl::transform([&](auto const day) {
               return concrete_trip{this, day};
             })  //
           | utl::iterable();
  }

  concrete_trip get_concrete_trip(generic_light_connection const& c) const {
    auto const e_it =
        std::find_if(begin(*edges_), end(*edges_),
                     [&](route_edge const& e) { return e->contains(c); });
    utl::verify(e_it != end(*edges_),
                "trip::get_concrete_trip(): edge not found in trip");

    auto const edge_idx = std::distance(begin(*edges_), e_it);
    auto const day = c.d_time().day() - day_offsets_.at(edge_idx);
    return concrete_trip{this, static_cast<day_idx_t>(day)};
  }

  bitfield const& traffic_days() const {
    return *edges_->front()->static_lcons().at(lcon_idx_).traffic_days_;
  }

  size_t ctrp_count() const { return traffic_days().count(); }

  bool operates_on_day(day_idx_t const day) const {
    if (day < 0U || day >= MAX_DAYS || edges_ == nullptr || edges_->empty()) {
      return false;
    } else if (edges_->front()->empty()) {
      return false;
    } else {
      return edges_->front()->static_lcons().front().traffic_days_->test(day);
    }
  }

  full_trip_id id_;
  ptr<mcd::vector<route_edge> const> edges_{nullptr};
  mcd::vector<day_idx_t> day_offsets_;
  lcon_idx_t lcon_idx_{0U};
  trip_debug dbg_;
  mcd::vector<uint32_t> stop_seq_numbers_;
};

}  // namespace motis
