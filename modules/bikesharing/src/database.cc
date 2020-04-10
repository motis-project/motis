#include "motis/bikesharing/database.h"

#include "lmdb/lmdb.hpp"

#include "utl/to_vec.h"

#include "motis/bikesharing/error.h"

using std::system_error;

namespace db = lmdb;
using namespace flatbuffers;

namespace motis::bikesharing {

constexpr auto kSummaryKey = "__summary";

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
    return txn.get(db, kSummaryKey).has_value();
  }

  virtual persistable_terminal get(std::string const& id) const {
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto db = txn.dbi_open();
    if (auto const r = txn.get(db, id); !r.has_value()) {
      throw system_error(error::terminal_not_found);
    } else {
      return persistable_terminal(*r);
    }
  }

  virtual void put(std::vector<persistable_terminal> const& terminals) {
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    for (auto const& t : terminals) {
      txn.put(db, t.get()->id()->str(), t.to_string());
    }
    txn.commit();
  }

  virtual bikesharing_summary get_summary() const {
    auto txn = db::txn{env_, db::txn_flags::RDONLY};
    auto db = txn.dbi_open();
    if (auto const r = txn.get(db, kSummaryKey); !r.has_value()) {
      throw system_error(error::database_error);
    } else {
      return bikesharing_summary(*r);
    }
  }

  virtual void put_summary(bikesharing_summary const& summary) {
    auto txn = db::txn{env_};
    auto db = txn.dbi_open();
    txn.put(db, kSummaryKey, summary.to_string());
    txn.commit();
  }

  db::env mutable env_;
};

struct inmemory_database : public database::database_impl {
  bool is_initialized() const override {
    auto it = store_.find(kSummaryKey);
    return it != end(store_);
  }

  persistable_terminal get(std::string const& id) const override {
    auto it = store_.find(id);
    if (it == end(store_)) {
      throw system_error(error::terminal_not_found);
    }
    return persistable_terminal(it->second);
  }

  void put(std::vector<persistable_terminal> const& terminals) override {
    for (auto const& t : terminals) {
      store_[t.get()->id()->str()] = t.to_string();
    }
  }

  bikesharing_summary get_summary() const override {
    auto it = store_.find(kSummaryKey);
    if (it == end(store_)) {
      throw system_error(error::database_not_initialized);
    }
    return bikesharing_summary(it->second);
  }

  void put_summary(bikesharing_summary const& summary) override {
    store_[kSummaryKey] = summary.to_string();
  }

  std::map<std::string, std::string> store_;
};

database::database(std::string const& path, size_t const max_size)
    : impl_(path == ":memory:" ? new inmemory_database()
                               : new database_impl(path, max_size)) {}

database::~database() = default;

bool database::is_initialized() const { return impl_->is_initialized(); }

persistable_terminal database::get(std::string const& id) const {
  return impl_->get(id);
}

void database::put(std::vector<persistable_terminal> const& terminals) {
  impl_->put(terminals);
}

bikesharing_summary database::get_summary() const {
  return impl_->get_summary();
}

void database::put_summary(bikesharing_summary const& summary) {
  impl_->put_summary(summary);
}

namespace detail {

Offset<Vector<Offset<Availability>>> create_availabilities(
    FlatBufferBuilder& b, hourly_availabilities const& availabilities) {
  std::vector<Offset<Availability>> vec;
  for (auto const& a : availabilities) {
    vec.push_back(CreateAvailability(b, a.average_, a.median_, a.minimum_,
                                     a.q90_, a.percent_reliable_));
  }
  return b.CreateVector(vec);
}

Offset<Vector<Offset<CloseLocation>>> create_close_locations(
    FlatBufferBuilder& b, std::vector<close_location> const& locations) {
  return b.CreateVector(utl::to_vec(locations, [&](auto&& l) {
    return CreateCloseLocation(b, b.CreateString(l.id_), l.duration_);
  }));
}

}  // namespace detail

persistable_terminal convert_terminal(
    terminal const& terminal, hourly_availabilities const& availabilities,
    std::vector<close_location> const& attached,
    std::vector<close_location> const& reachable) {
  FlatBufferBuilder b;
  b.Finish(CreateTerminal(b, b.CreateString(terminal.uid_), terminal.lat_,
                          terminal.lng_, b.CreateString(terminal.name_),
                          detail::create_availabilities(b, availabilities),
                          detail::create_close_locations(b, attached),
                          detail::create_close_locations(b, reachable)));
  return persistable_terminal(std::move(b));
}

bikesharing_summary make_summary(std::vector<terminal> const& terminals) {
  FlatBufferBuilder b;
  b.Finish(CreateSummary(
      b, b.CreateVector(utl::to_vec(terminals, [&](auto&& terminal) {
        return CreateTerminalLocation(b, b.CreateString(terminal.uid_),
                                      terminal.lat_, terminal.lng_);
      }))));
  return bikesharing_summary(std::move(b));
}

}  // namespace motis::bikesharing
