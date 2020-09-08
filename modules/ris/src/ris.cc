#include "motis/ris/ris.h"

#include <cstdint>
#include <atomic>
#include <limits>
#include <optional>

#include "boost/filesystem.hpp"

#include "utl/concat.h"
#include "utl/parser/file.h"

#include "conf/date_time.h"

#include "tar/file_reader.h"
#include "tar/tar_reader.h"
#include "tar/zstd_reader.h"

#include "utl/verify.h"

#include "lmdb/lmdb.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/print_trip.h"
#include "motis/module/context/get_schedule.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/ris/ris_message.h"
#include "motis/ris/risml/risml_parser.h"
#include "motis/ris/zip_reader.h"

#ifdef GetMessage
#undef GetMessage
#endif

namespace fs = boost::filesystem;
namespace db = lmdb;
using namespace motis::module;
using namespace motis::logging;
using tar::file_reader;
using tar::tar_reader;
using tar::zstd_reader;

namespace motis::ris {

// stores the list of files that were already parsed
// key: path
// value: empty
constexpr auto const FILE_DB = "FILE_DB";

// messages, no specific order (unique id)
// key: timestamp
// value: buffer of messages:
//        2 bytes message size, {message size} bytes message
constexpr auto const MSG_DB = "MSG_DB";

// index for every day referenced by any message
// key: day.begin (unix timestamp)
// value: smallest message timestamp from MSG_DB that has
//        earliest <= day.end && latest >= day.begin
constexpr auto const MIN_DAY_DB = "MIN_DAY_DB";

// index for every day referenced by any message
// key: day.begin (unix timestamp)
// value: largest message timestamp from MSG_DB that has
//        earliest <= day.end && latest >= day.begin
constexpr auto const MAX_DAY_DB = "MAX_DAY_DB";

constexpr auto const BATCH_SIZE = time_t{3600};

constexpr auto const WRITE_MSG_BUF_MAX_SIZE = 50000;

template <typename T>
constexpr T floor(T const i, T const multiple) {
  return (i / multiple) * multiple;
}

template <typename T>
constexpr T ceil(T const i, T const multiple) {
  return ((i - 1) / multiple) * multiple + multiple;
}

using size_type = uint32_t;
constexpr auto const SIZE_TYPE_SIZE = sizeof(size_type);

constexpr time_t day(time_t t) { return (t / SECONDS_A_DAY) * SECONDS_A_DAY; }
constexpr time_t next_day(time_t t) { return day(t) + SECONDS_A_DAY; }

template <typename Fn>
inline void for_each_day(ris_message const& m, Fn&& f) {
  auto const last = next_day(m.latest_);
  for (auto d = day(m.earliest_); d != last; d += SECONDS_A_DAY) {
    f(d);
  }
}

time_t to_time_t(std::string_view s) {
  return *reinterpret_cast<time_t const*>(s.data());
}

std::string_view from_time_t(time_t const& t) {
  return {reinterpret_cast<char const*>(&t), sizeof(t)};
}

struct ris::impl {
  void init() {
    if (!gtfs_trip_ids_path_.empty()) {
      read_gtfs_trip_ids();
    }

    if (clear_db_) {
      LOG(info) << "clearing database path " << db_path_;
      fs::remove_all(db_path_);
    }

    env_.set_maxdbs(4);
    env_.set_mapsize(db_max_size_);
    env_.open(db_path_.c_str(),
              lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOTLS);

    db::txn t{env_};
    t.dbi_open(FILE_DB, db::dbi_flags::CREATE);
    t.dbi_open(MSG_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MIN_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MAX_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.commit();

    if (fs::exists(input_)) {
      LOG(warn) << "parsing " << input_;
      // parse_parallel(input_, null_pub_);
      if (instant_forward_) {
        publisher pub;
        parse_sequential(input_, pub);
      } else {
        parse_sequential(input_, null_pub_);
      }
    } else {
      LOG(warn) << input_ << " does not exist";
    }

    if (init_time_ != 0) {
      forward(init_time_);
    }
  }

  void read_gtfs_trip_ids() const {
    auto& sched = get_schedule();
    auto const trips = utl::file{gtfs_trip_ids_path_.c_str(), "r"}.content();
    auto const trips_msg =
        make_msg(trips.data(), trips.size(), DEFAULT_FBS_MAX_DEPTH,
                 std::numeric_limits<std::uint32_t>::max());
    for (auto const& id : *motis_content(RISGTFSRTMapping, trips_msg)->ids()) {
      try {
        sched.gtfs_trip_ids_.emplace(
            gtfs_trip_id{id->id()->str(), static_cast<std::time_t>(id->day())},
            from_fbs(sched, id->trip(), true));
      } catch (...) {
        std::cout << to_extern_trip(id->trip()) << "\n";
      }
    }
    LOG(info) << sched.gtfs_trip_ids_.size() << "/"
              << motis_content(RISGTFSRTMapping, trips_msg)->ids()->size()
              << " imported";
  }

  msg_ptr upload(msg_ptr const& msg) {
    auto const content = motis_content(HTTPRequest, msg)->content();
    auto& sched = get_schedule();
    publisher pub;
    auto risml_fn = [](std::string_view s,
                       std::function<void(ris_message &&)> const& cb) {
      risml::risml_parser::to_ris_message(s, cb);
    };

    write_to_db(zip_reader{content->c_str(), content->size()}, risml_fn, pub);
    sched.system_time_ = pub.max_timestamp_;
    sched.last_update_timestamp_ = std::time(nullptr);
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
    return {};
  }

  msg_ptr read(msg_ptr const&) {
    auto& sched = get_schedule();
    publisher pub;
    parse_sequential(input_, pub);
    sched.system_time_ = pub.max_timestamp_;
    sched.last_update_timestamp_ = std::time(nullptr);
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
    return {};
  }

  msg_ptr forward(msg_ptr const& msg) {
    forward(motis_content(RISForwardTimeRequest, msg)->new_time());
    return {};
  }

  msg_ptr purge(msg_ptr const& msg) {
    auto const until =
        static_cast<time_t>(motis_content(RISPurgeRequest, msg)->until());

    auto t = db::txn{env_};
    auto db = t.dbi_open(MSG_DB);
    auto c = db::cursor{t, db};
    auto bucket = c.get(db::cursor_op::SET_RANGE, until);
    while (bucket) {
      if (bucket->first <= until) {
        c.del();
      }
      bucket = c.get(db::cursor_op::PREV, 0);
    }
    t.commit();
    c.reset();

    return {};
  }

  std::string gtfs_trip_ids_path_;
  std::string db_path_{"ris.mdb"};
  std::string input_{"ris"};
  conf::holder<std::time_t> init_time_{0};
  bool clear_db_ = false;
  size_t db_max_size_{static_cast<size_t>(1024) * 1024 * 1024 * 512};
  bool instant_forward_{false};
  risml::risml_parser risml_parser_;
  gtfsrt::gtfsrt_parser gtfsrt_parser_;

  impl() = default;
  ~impl() = default;
  impl(impl const&) = delete;
  impl& operator=(impl const&) = delete;
  impl(impl&&) = delete;
  impl& operator=(impl&&) = delete;

private:
  struct publisher {
    publisher() = default;
    publisher(publisher&&) = delete;
    publisher(publisher const&) = delete;
    publisher& operator=(publisher&&) = delete;
    publisher& operator=(publisher const&) = delete;

    ~publisher() { flush(); }

    void flush() {
      if (offsets_.empty()) {
        return;
      }

      fbb_.create_and_finish(
          MsgContent_RISBatch,
          CreateRISBatch(fbb_, fbb_.CreateVector(offsets_)).Union(),
          "/ris/messages");

      auto msg = make_msg(fbb_);
      fbb_.Clear();
      offsets_.clear();

      ctx::await_all(motis_publish(msg));
    }

    void add(uint8_t const* ptr, size_t const size) {
      max_timestamp_ = std::max(
          max_timestamp_,
          static_cast<time_t>(
              flatbuffers::GetRoot<Message>(reinterpret_cast<void const*>(ptr))
                  ->timestamp()));
      offsets_.push_back(
          CreateMessageHolder(fbb_, fbb_.CreateVector(ptr, size)));
    }

    size_t size() const { return offsets_.size(); }

    message_creator fbb_;
    std::vector<flatbuffers::Offset<MessageHolder>> offsets_;
    time_t max_timestamp_ = 0;
  };

  struct null_publisher {
    void flush() {}
    void add(uint8_t const*, size_t const) {}
    size_t size() const { return 0; }  // NOLINT
    time_t max_timestamp_ = 0;
  } null_pub_;

  void forward(time_t const to) {
    auto const& sched = get_schedule();
    auto const first_schedule_event_day = floor(
        sched.first_event_schedule_time_, static_cast<time_t>(SECONDS_A_DAY));
    auto const last_schedule_event_day = ceil(
        sched.last_event_schedule_time_, static_cast<time_t>(SECONDS_A_DAY));
    auto const min_timestamp =
        get_min_timestamp(first_schedule_event_day, last_schedule_event_day);
    if (min_timestamp) {
      forward(std::max(*min_timestamp, sched.system_time_ + 1), to);
    } else {
      LOG(info) << "ris database has no relevant data";
    }
  }

  void forward(time_t const from, time_t const to) {
    LOG(info) << "forwarding from " << logging::time(from) << " to "
              << logging::time(to);

    auto t = db::txn{env_, db::txn_flags::RDONLY};
    auto db = t.dbi_open(MSG_DB);
    auto c = db::cursor{t, db};
    auto bucket = c.get(db::cursor_op::SET_RANGE, from);
    auto batch_begin = bucket ? bucket->first : 0;
    publisher pub;
    auto const& sched = get_schedule();
    while (true) {
      if (!bucket) {
        LOG(info) << "end of db reached";
        break;
      }

      auto const& [timestamp, msgs] = *bucket;
      if (timestamp > to) {
        break;
      }

      auto ptr = msgs.data();
      auto const end = ptr + msgs.size();
      while (ptr < end) {
        size_type size = 0;
        std::memcpy(&size, ptr, SIZE_TYPE_SIZE);
        ptr += SIZE_TYPE_SIZE;

        if (size == 0) {
          continue;
        }

        utl::verify(ptr + size <= end, "ris: ptr + size > end");

        if (auto const msg = GetMessage(ptr);
            msg->timestamp() <= to && msg->timestamp() >= from &&
            msg->earliest() <= sched.last_event_schedule_time_ &&
            msg->latest() >= sched.first_event_schedule_time_) {
          pub.add(reinterpret_cast<uint8_t const*>(ptr), size);
        }

        ptr += size;
      }

      if (timestamp - batch_begin > BATCH_SIZE) {
        LOG(logging::info) << "(" << logging::time(batch_begin) << " - "
                           << logging::time(batch_begin + BATCH_SIZE)
                           << ") flushing " << pub.size() << " messages";
        pub.flush();
        batch_begin = timestamp;
      }

      bucket = c.get(db::cursor_op::NEXT, 0);
    }

    pub.flush();
    get_schedule().system_time_ = to;
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
  }

  std::optional<time_t> get_min_timestamp(time_t const from_day,
                                          time_t const to_day) {
    utl::verify(from_day % SECONDS_A_DAY == 0, "from not a day");
    utl::verify(to_day % SECONDS_A_DAY == 0, "to not a day");

    constexpr auto const max = std::numeric_limits<time_t>::max();

    auto min = max;
    auto t = db::txn{env_, db::txn_flags::RDONLY};
    auto db = t.dbi_open(MIN_DAY_DB);
    for (auto d = from_day; d != to_day; d += SECONDS_A_DAY) {
      auto const r = t.get(db, d);
      if (r) {
        min = std::min(min, to_time_t(*r));
      }
    }

    return min != max ? std::make_optional(min) : std::nullopt;
  }

  enum class file_type { NONE, ZST, ZIP, XML, PROTOBUF };

  static file_type get_file_type(fs::path const& p) {
    if (p.extension() == ".zst") {
      return file_type::ZST;
    } else if (p.extension() == ".zip") {
      return file_type::ZIP;
    } else if (p.extension() == ".xml") {
      return file_type::XML;
    } else if (p.extension() == ".pb") {
      return file_type::PROTOBUF;
    } else {
      return file_type::NONE;
    }
  }

  std::vector<std::tuple<time_t, fs::path, file_type>> collect_files(
      fs::path const& p) {
    if (fs::is_regular_file(p)) {
      if (auto const t = get_file_type(p);
          t != file_type::NONE && !is_known_file(p)) {
        return {std::make_tuple(fs::last_write_time(p), p, t)};
      }
    } else if (fs::is_directory(p)) {
      std::vector<std::tuple<time_t, fs::path, file_type>> files;
      for (auto const& entry : fs::directory_iterator(p)) {
        utl::concat(files, collect_files(entry));
      }
      std::sort(begin(files), end(files));
      return files;
    }
    return {};
  }

  template <typename Publisher>
  void parse_parallel(fs::path const& p, Publisher& pub) {
    ctx::await_all(utl::to_vec(
        collect_files(fs::canonical(p, p.root_path())), [&](auto&& e) {
          return spawn_job_void([e, this, &pub]() {
            write_to_db(std::get<1>(e), std::get<2>(e), pub);
          });
        }));
    env_.force_sync();
  }

  template <typename Publisher>
  void parse_sequential(fs::path const& p, Publisher& pub) {
    for (auto const& [t, path, type] :
         collect_files(fs::canonical(p, p.root_path()))) {
      ((void)(t));
      write_to_db(path, type, pub);
      if (instant_forward_) {
        get_schedule().system_time_ = pub.max_timestamp_;
        get_schedule().last_update_timestamp_ = std::time(nullptr);
        try {
          ctx::await_all(
              motis_publish(make_no_msg("/ris/system_time_changed")));
        } catch (std::system_error& e) {
          LOG(info) << e.what();
        }
      }
    }
    env_.force_sync();
  }

  bool is_known_file(fs::path const& p) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(FILE_DB);
    return t.get(db, p.generic_string()).has_value();
  }

  void add_to_known_files(fs::path const& p) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(FILE_DB);
    t.put(db, p.generic_string(), "");
    t.commit();
  }

  template <typename Publisher>
  void write_to_db(fs::path const& p, file_type const type, Publisher& pub) {
    using tar_zst = tar_reader<zstd_reader>;
    auto const& cp = p.generic_string();

    auto risml_fn = [this](std::string_view s,
                           std::function<void(ris_message &&)> const& cb) {
      risml_parser_.to_ris_message(s, cb);
    };
    auto gtfsrt_fn = [this](std::string_view s,
                            std::function<void(ris_message &&)> const& cb) {
      gtfsrt_parser_.to_ris_message(s, cb);
    };
    try {
      switch (type) {
        case file_type::ZST:
          write_to_db(tar_zst(zstd_reader(cp.c_str())), risml_fn, pub);
          break;
        case file_type::ZIP:
          write_to_db(zip_reader(cp.c_str()), risml_fn, pub);
          break;
        case file_type::XML:
          write_to_db(file_reader(cp.c_str()), risml_fn, pub);
          break;
        case file_type::PROTOBUF:
          write_to_db(file_reader(cp.c_str()), gtfsrt_fn, pub);
          break;
        default: assert(false);
      }
    } catch (...) {
      LOG(logging::error) << "failed to read " << p;
    }
    add_to_known_files(p);
  }

  template <typename Reader, typename ParserFn, typename Publisher>
  void write_to_db(Reader&& reader, ParserFn parser_fn, Publisher& pub) {
    std::map<time_t /* d.b */, time_t /* min(t) : e <= d.e && l >= d.b */> min;
    std::map<time_t /* d.b */, time_t /* max(t) : e <= d.e && l >= d.b */> max;
    std::map<time_t /* tout */, std::vector<char>> buf;
    auto buf_msg_count = 0U;

    auto flush_to_db = [&]() {
      if (buf.empty()) {
        return;
      }

      std::lock_guard<std::mutex> lock{merge_mutex_};

      auto t = db::txn{env_};
      auto db = t.dbi_open(MSG_DB);
      auto c = db::cursor{t, db};

      for (auto& [timestamp, entry] : buf) {
        if (auto const v = c.get(lmdb::cursor_op::SET_RANGE, timestamp);
            v && v->first == timestamp) {
          entry.insert(end(entry), begin(v->second), end(v->second));
        }
        c.put(timestamp, std::string_view{&entry[0], entry.size()});
      }

      c.commit();
      t.commit();

      buf.clear();
    };

    auto write = [&](ris_message&& m) {
      if (buf_msg_count++ > WRITE_MSG_BUF_MAX_SIZE) {
        flush_to_db();
        buf_msg_count = 0;
      }

      auto& buf_val = buf[m.timestamp_];
      auto const base = buf_val.size();
      buf_val.resize(buf_val.size() + SIZE_TYPE_SIZE + m.size());

      auto const msg_size = static_cast<size_type>(m.size());
      std::memcpy(&buf_val[0] + base, &msg_size, SIZE_TYPE_SIZE);
      std::memcpy(&buf_val[0] + base + SIZE_TYPE_SIZE, m.data(), m.size());

      pub.add(m.data(), m.size());

      for_each_day(m, [&](time_t const d) {
        if (auto it = min.lower_bound(d); it != end(min) && it->first == d) {
          it->second = std::min(it->second, m.timestamp_);
        } else {
          min.emplace_hint(it, d, m.timestamp_);
        }

        if (auto it = max.lower_bound(d); it != end(max) && it->first == d) {
          it->second = std::max(it->second, m.timestamp_);
        } else {
          max.emplace_hint(it, d, m.timestamp_);
        }
      });
    };

    auto parse = std::forward<ParserFn>(parser_fn);

    std::optional<std::string_view> reader_content;
    while ((reader_content = reader.read())) {
      parse(*reader_content, [&](ris_message&& m) { write(std::move(m)); });
    }

    flush_to_db();
    update_min_max(min, max);
    pub.flush();
  }

  void update_min_max(std::map<time_t, time_t> const& min,
                      std::map<time_t, time_t> const& max) {
    std::lock_guard<std::mutex> lock{min_max_mutex_};

    auto t = db::txn{env_};
    auto min_db = t.dbi_open(MIN_DAY_DB);
    auto max_db = t.dbi_open(MAX_DAY_DB);

    for (auto const [day, min_timestamp] : min) {
      auto smallest = min_timestamp;
      if (auto entry = t.get(min_db, day); entry) {
        smallest = std::min(smallest, to_time_t(*entry));
      }
      t.put(min_db, day, from_time_t(smallest));
    }

    for (auto const [day, max_timestamp] : max) {
      auto largest = max_timestamp;
      if (auto entry = t.get(max_db, day); entry) {
        largest = std::max(largest, to_time_t(*entry));
      }
      t.put(max_db, day, from_time_t(largest));
    }

    t.commit();
  }

  db::env env_;
  std::atomic<uint64_t> next_msg_id_{0};
  std::mutex min_max_mutex_;
  std::mutex merge_mutex_;
};

ris::ris() : module("RIS", "ris"), impl_(std::make_unique<impl>()) {
  param(impl_->gtfs_trip_ids_path_, "gtfs_trip_ids",
        "path to GTFS trip ids file");
  param(impl_->db_path_, "db", "ris database path");
  param(impl_->input_, "input", "ris input (folder or risml)");
  param(impl_->db_max_size_, "db_max_size", "virtual memory map size");
  param(impl_->init_time_, "init_time", "initial forward time");
  param(impl_->clear_db_, "clear_db", "clean db before init");
  param(impl_->instant_forward_, "instant_forward",
        "automatically forward after every file during read");
  param(impl_->gtfsrt_parser_.is_addition_skip_allowed_,
        "gtfsrt.is_addition_skip_allowed", "allow skips on additional trips");
}

ris::~ris() = default;

void ris::init(motis::module::registry& r) {
  r.subscribe(
      "/init", [this]() { impl_->init(); }, ctx::access_t::WRITE);
  r.register_op(
      "/ris/upload", [this](auto&& m) { return impl_->upload(m); },
      ctx::access_t::WRITE);
  r.register_op(
      "/ris/forward", [this](auto&& m) { return impl_->forward(m); },
      ctx::access_t::WRITE);
  r.register_op(
      "/ris/read", [this](auto&& m) { return impl_->read(m); },
      ctx::access_t::WRITE);
  r.register_op(
      "/ris/purge", [this](auto&& m) { return impl_->purge(m); },
      ctx::access_t::WRITE);
  r.register_op(
      "/ris/write_gtfs_trip_ids",
      [](auto&&) {
        message_creator fbb;
        auto const& sched = get_schedule();
        fbb.create_and_finish(
            MsgContent_RISGTFSRTMapping,
            CreateRISGTFSRTMapping(
                fbb,
                fbb.CreateVector(utl::to_vec(
                    sched.gtfs_trip_ids_,
                    [&](mcd::pair<gtfs_trip_id, ptr<trip const>> const& id) {
                      // SBB HRD data uses eva numbers
                      // GTFS uses ${eva number}:0:${track}
                      // To use SBB GTFS station indices in HRD:
                      // -> cut and export the eva number (the part until ':')
                      auto const cut = [](std::string const& s) {
                        auto const i = s.find_first_of(':');
                        return i != std::string::npos ? s.substr(0, i) : s;
                      };
                      auto const& p = id.second->id_.primary_;
                      auto const& s = id.second->id_.secondary_;
                      return CreateGTFSID(
                          fbb, fbb.CreateString(id.first.trip_id_),
                          id.first.start_date_,
                          CreateTripId(
                              fbb,
                              fbb.CreateString(
                                  cut(sched.stations_.at(p.station_id_)
                                          ->eva_nr_.str())),
                              p.train_nr_, motis_to_unixtime(sched, p.time_),
                              fbb.CreateString(
                                  cut(sched.stations_.at(s.target_station_id_)
                                          ->eva_nr_.str())),
                              motis_to_unixtime(sched, s.target_time_),
                              fbb.CreateString(s.line_id_)));
                    })))
                .Union());
        auto const msg = make_msg(fbb);
        utl::file{"gtfs_trips.raw", "w"}.write(msg->data(), msg->size());
        return nullptr;
      },
      ctx::access_t::READ);
}

}  // namespace motis::ris
