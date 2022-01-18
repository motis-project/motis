#include "motis/ridesharing/database.h"

#include "motis/core/common/logging.h"

#include "motis/ridesharing/error.h"

#include <map>

#include "lmdb/lmdb.hpp"
#include "utl/to_vec.h"

using std::system_error;

namespace db = lmdb;
using namespace flatbuffers;

namespace motis::ridesharing {

constexpr auto kRoutingTableKey = "table__";
constexpr auto kHashKey = "rthash__";

struct database::database_impl {
  database_impl() = default;
  virtual ~database_impl() = default;

  database_impl(database_impl const&) = delete;
  database_impl& operator=(database_impl const&) = delete;

  database_impl(database_impl&&) = default;
  database_impl& operator=(database_impl&&) = default;

  explicit database_impl(std::string const& path, size_t const max_size) {
    env_.set_maxdbs(1);
    env_.set_mapsize(max_size);
    env_.open(path.c_str(), db::env_open_flags::NOSUBDIR |
                                db::env_open_flags::NOLOCK |
                                db::env_open_flags::NOTLS);
  }

  virtual bool is_initialized() const {
    if (!env_.is_open()) {
      return false;
    }
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    return txn.get(db, kRoutingTableKey).has_value();
  }

  virtual std::optional<lift> get_lift(lift_key const& key) const {
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto db = txn.dbi_open();
    auto const r = txn.get(db, key.to_string());
    if (!r.has_value()) {
      return {};
    } else {
      auto const li = persistable_lift(*r);
      return {from_db(li.get())};
    }
  }

  virtual void put_lift(persistable_lift const& lift, lift_key const& key) {
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    txn.put(db, key.to_string(), lift.to_string());
    txn.commit();
  }

  virtual bool remove_lift(lift_key const& key) {
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    txn.del(db, key.to_string());
    txn.commit();
    return true;
  }

  virtual std::vector<lift> get_lifts() const {
    auto lifts = std::vector<lift>{};
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto dbi = txn.dbi_open();
    auto cur = db::cursor{txn, dbi};
    auto bucket = cur.get(db::cursor_op::FIRST);
    while (bucket) {
      auto const val = bucket.value();
      if (val.first == kRoutingTableKey || val.first == kHashKey) {
        break;
      }
      auto const li = persistable_lift(val.second);
      lifts.push_back(from_db(li.get()));
      bucket = cur.get(db::cursor_op::NEXT);
    }
    return lifts;
  }

  virtual long get_station_hashcode() const {
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto db = txn.dbi_open();
    if (auto const r = txn.get(db, kHashKey); !r.has_value()) {
      throw system_error(error::database_error);
    } else {
      return std::stol(std::string{r.value()});
    }
  }

  virtual std::vector<std::vector<routing_result>> get_routing_table() const {
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto db = txn.dbi_open();
    if (auto const r = txn.get(db, kRoutingTableKey); !r.has_value()) {
      throw system_error(error::database_error);
    } else {
      auto const rt = persistable_routing_table(*r);
      return utl::to_vec(*rt.get()->routing_matrix(), [](auto const& rr_row) {
        return utl::to_vec(*rr_row->costs(), [](auto const& rr) {
          return routing_result{rr->duration(), rr->distance()};
        });
      });
    }
  }

  virtual void put_routing_table(persistable_routing_table const& routing_table,
                                 long const hashcode) {
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    txn.put(db, kRoutingTableKey, routing_table.to_string());
    txn.put(db, kHashKey, std::to_string(hashcode));
    txn.commit();
  }

  db::env mutable env_;
};

struct inmemory_database : public database::database_impl {
  bool is_initialized() const override {
    auto it = store_.find(kRoutingTableKey);
    return it != end(store_);
  }

  std::optional<lift> get_lift(lift_key const& key) const override {
    auto it = store_.find(key.to_string());
    if (it == end(store_)) {
      return {};
    }
    auto const li = persistable_lift(it->second);
    return {from_db(li.get())};
  }

  void put_lift(persistable_lift const& lift, lift_key const& key) override {
    store_[key.to_string()] = lift.to_string();
  }

  bool remove_lift(lift_key const& key) override {
    auto it = store_.find(key.to_string());
    if (it != end(store_)) {
      store_.erase(it);
      return true;
    }
    return false;
  }

  std::vector<lift> get_lifts() const override {
    auto lifts = std::vector<lift>{};
    for (auto const& entry : store_) {
      if (entry.first != kRoutingTableKey && entry.first != kHashKey) {
        auto const li = persistable_lift(entry.second);
        lifts.push_back(from_db(li.get()));
      }
    }
    return lifts;
  }

  long get_station_hashcode() const override {
    auto it = store_.find(kHashKey);
    if (it == end(store_)) {
      throw system_error(error::database_not_initialized);
    }
    return std::stol(it->second);
  }

  std::vector<std::vector<routing_result>> get_routing_table() const override {
    auto it = store_.find(kRoutingTableKey);
    if (it == end(store_)) {
      throw system_error(error::database_not_initialized);
    }
    auto const rt = persistable_routing_table(it->second);
    return utl::to_vec(*rt.get()->routing_matrix(), [](auto const& rr_row) {
      return utl::to_vec(*rr_row->costs(), [](auto const& rr) {
        return routing_result{rr->duration(), rr->distance()};
      });
    });
  }

  void put_routing_table(persistable_routing_table const& routing_table,
                         long const hashcode) override {
    store_[kRoutingTableKey] = routing_table.to_string();
    store_[kHashKey] = std::to_string(hashcode);
  }

  std::map<std::string, std::string> store_;
};

database::database(std::string const& path, size_t const max_size)
    : impl_(path == ":memory:" ? new inmemory_database()
                               : new database_impl(path, max_size)) {}

database::~database() = default;

bool database::is_initialized() const { return impl_->is_initialized(); }

std::optional<lift> database::get_lift(lift_key const& key) const {
  return impl_->get_lift(key);
}
void database::put_lift(persistable_lift const& lift, lift_key const& key) {
  impl_->put_lift(lift, key);
}
bool database::remove_lift(lift_key const& key) {
  return impl_->remove_lift(key);
}
std::vector<lift> database::get_lifts() const { return impl_->get_lifts(); }
std::vector<std::vector<routing_result>> database::get_routing_table() const {
  return impl_->get_routing_table();
}
void database::put_routing_table(persistable_routing_table const& routing_table,
                                 long const hashcode) {
  impl_->put_routing_table(routing_table, hashcode);
}
long database::get_station_hashcode() const {
  return impl_->get_station_hashcode();
}

lift from_db(DBLift const* db_lift) {
  auto const waypoints =
      utl::to_vec(*db_lift->waypoints(), [](auto const& loc) {
        return geo::latlng{loc->lat(), loc->lng()};
      });
  auto const rrs = utl::to_vec(*db_lift->routing_results(), [](auto const& rr) {
    return routing_result{rr->duration(), rr->distance()};
  });

  auto const passengers =
      utl::to_vec(*db_lift->passengers(), [](auto const& p) {
        return passenger{p->passenger_id(),
                         {p->from()->lat(), p->from()->lng()},
                         {p->to()->lat(), p->to()->lng()},
                         p->price(),
                         p->required_arrival(),
                         p->passenger_count()};
      });
  auto li = lift{waypoints,
                 rrs,
                 db_lift->max_total_time(),
                 db_lift->lift_start_time(),
                 db_lift->driver_id(),
                 db_lift->max_passengers(),
                 passengers};

  li.to_routings_ = utl::to_vec(*db_lift->to_routings(), [](auto const& row) {
    auto res = std::unordered_map<unsigned, routing_result>{};
    for (auto const& entry : *row->routings()) {
      res.insert({entry->station(), {entry->duration(), entry->distance()}});
    }
    return res;
  });
  li.from_routings_ = utl::to_vec(*db_lift->from_routings(), [](auto const&
                                                                    row) {
    auto res = std::unordered_map<unsigned, routing_result>{};
    for (auto const& entry : *row->routings()) {
      res.insert({entry->station(), {entry->duration(), entry->distance()}});
    }
    return res;
  });
  return li;
}

persistable_lift make_db_lift(lift const& l) {
  FlatBufferBuilder b;

  std::vector<Offset<DBAcceptableStationsMapEntry>> fbs_from_to;
  /*for (auto& entry : l.from_to_) {
    fbs_from_to.push_back(CreateDBAcceptableStationsMapEntry(
        b, entry.first, entry.second.from_leg_, entry.second.to_leg_,
        b.CreateVector(entry.second.stations_)));
  }*/

  std::vector<LatLng> fbs_waypoints = utl::to_vec(l.waypoints_, [](auto& loc) {
    return LatLng{loc.lat_, loc.lng_};
  });

  std::vector<Offset<DBRoutingResult>> fbs_routing_results;
  for (auto& rr : l.rrs_) {
    fbs_routing_results.push_back(
        CreateDBRoutingResult(b, rr.duration_, rr.distance_));
  }

  std::vector<Offset<DBPassenger>> fbs_passengers;
  for (auto& p : l.passengers_) {
    LatLng piu{p.pick_up_.lat_, p.pick_up_.lng_};
    LatLng dro{p.drop_off_.lat_, p.drop_off_.lng_};
    fbs_passengers.push_back(CreateDBPassenger(b, p.passenger_id_, &piu, &dro,
                                               p.price_, p.required_arrival_,
                                               p.passenger_count_));
  }

  auto const fbs_to_routings =
      utl::to_vec(l.to_routings_, [&](auto const& row) {
        return CreateDBStationRouting(
            b, b.CreateVector(utl::to_vec(row, [&](auto const& entry) {
              return CreateDBRoutingEntry(b, entry.first,
                                          entry.second.duration_,
                                          entry.second.distance_);
            })));
      });

  auto const fbs_from_routings =
      utl::to_vec(l.from_routings_, [&](auto const& row) {
        return CreateDBStationRouting(
            b, b.CreateVector(utl::to_vec(row, [&](auto const& entry) {
              return CreateDBRoutingEntry(b, entry.first,
                                          entry.second.duration_,
                                          entry.second.distance_);
            })));
      });

  /*for (auto& waypoint : ) {
     std::vector<Offset<DBRoutingResult>> to_rrs;
     for (auto& to_rr : waypoint) {
       to_rrs.push_back(
           CreateDBRoutingResult(b, to_rr.duration_, to_rr.distance_));
     }
     fbs_to_routings.push_back(
         CreateDBStationRouting(b, b.CreateVector(to_rrs)));
   }

   std::vector<Offset<DBStationRouting>> fbs_from_routings;*/
  /*for (auto& waypoint : l.from_routings_) {
    std::vector<Offset<DBRoutingResult>> from_rrs;
    for (auto& from_rr : waypoint) {
      from_rrs.push_back(
          CreateDBRoutingResult(b, from_rr.duration_, from_rr.distance_));
    }
    fbs_from_routings.push_back(
        CreateDBStationRouting(b, b.CreateVector(from_rrs)));
  }*/

  b.Finish(CreateDBLift(
      b, l.driver_id_, b.CreateVectorOfStructs(fbs_waypoints),
      b.CreateVector(fbs_routing_results), b.CreateVector(fbs_passengers),
      b.CreateVector(fbs_to_routings), b.CreateVector(fbs_from_routings),
      l.max_total_duration_, l.max_passengers_, l.t_));
  return persistable_lift(std::move(b));
}

persistable_routing_table make_routing_table(
    std::vector<std::vector<routing_result>> const& routing_matrix) {
  FlatBufferBuilder b;
  auto const fbs_routing_matrix =
      utl::to_vec(routing_matrix, [&](auto const& rr_row) {
        return CreateRoutingRow(
            b, b.CreateVector(utl::to_vec(rr_row, [&](auto& row) {
              return CreateCost(b, row.duration_, row.distance_);
            })));
      });
  b.Finish(CreateRoutingTable(b, b.CreateVector(fbs_routing_matrix)));
  return persistable_routing_table(std::move(b));
}

}  // namespace motis::ridesharing
