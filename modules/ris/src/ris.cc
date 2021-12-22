#include "motis/ris/ris.h"

#include <cstdint>
#include <atomic>
#include <limits>
#include <optional>

#include "boost/algorithm/string/predicate.hpp"
#include "boost/filesystem.hpp"

#include "utl/concat.h"
#include "utl/parser/file.h"
#include "utl/read_file.h"

#include "conf/date_time.h"

#include "tar/file_reader.h"
#include "tar/tar_reader.h"
#include "tar/zstd_reader.h"

#include "utl/verify.h"

#include "lmdb/lmdb.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/print_trip.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/ris/gtfs-rt/util.h"
#include "motis/ris/ribasis/ribasis_parser.h"
#include "motis/ris/ris_message.h"
#include "motis/ris/risml/risml_parser.h"
#include "motis/ris/string_view_reader.h"
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

constexpr auto const BATCH_SIZE = unixtime{3600};

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

constexpr unixtime day(unixtime t) {
  return (t / SECONDS_A_DAY) * SECONDS_A_DAY;
}
constexpr unixtime next_day(unixtime t) { return day(t) + SECONDS_A_DAY; }

template <typename Fn>
inline void for_each_day(ris_message const& m, Fn&& f) {
  auto const last = next_day(m.latest_);
  for (auto d = day(m.earliest_); d != last; d += SECONDS_A_DAY) {
    f(d);
  }
}

unixtime to_unixtime(std::string_view s) {
  return *reinterpret_cast<unixtime const*>(s.data());
}

std::string_view from_unixtime(unixtime const& t) {
  return {reinterpret_cast<char const*>(&t), sizeof(t)};
}

struct ris::impl {
  /**
   * Extracts the station prefix + directory path from a string formed
   * "${station-prefix}:${input-directory-path}"
   *
   * Contains an optional tag which is used as station id prefix to match
   * stations in multi-schedule mode and the path to the directory.
   *
   * In single-timetable mode, the tag should be omitted.
   */
  struct input {
    explicit input(std::string const& in) : input{split(in)} {}

    fs::path const& path() const { return path_; }
    std::string const& tag() const { return tag_; }
    gtfsrt::knowledge_context& gtfs_knowledge() { return gtfs_knowledge_; }

  private:
    explicit input(std::pair<fs::path, std::string>&& path_and_tag)
        : path_{std::move(path_and_tag.first)},
          tag_{path_and_tag.second},
          gtfs_knowledge_{path_and_tag.second} {}

    static std::pair<fs::path, std::string> split(std::string const& in) {
      std::string tag;
      if (auto const colon_pos = in.find(':'); colon_pos != std::string::npos) {
        tag = in.substr(0, colon_pos);
        tag = tag.empty() ? "" : tag + "_";
        return std::pair{fs::path{in.substr(colon_pos + 1)}, tag};
      } else {
        return std::pair{fs::path{in}, tag};
      }
    }

    fs::path path_;
    std::string tag_;

    gtfsrt::knowledge_context gtfs_knowledge_;
  };

  impl(schedule& sched, config const& c) : sched_{sched}, config_{c} {}

  void init() {
    inputs_ = utl::to_vec(config_.input_,
                          [](std::string const& in) { return input{in}; });

    if (config_.clear_db_ && fs::exists(config_.db_path_)) {
      LOG(info) << "clearing database path " << config_.db_path_;
      fs::remove_all(config_.db_path_);
    }

    env_.set_maxdbs(4);
    env_.set_mapsize(config_.db_max_size_);
    env_.open(config_.db_path_.c_str(),
              lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOTLS);

    db::txn t{env_};
    t.dbi_open(FILE_DB, db::dbi_flags::CREATE);
    t.dbi_open(MSG_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MIN_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MAX_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.commit();

    for (auto& in : inputs_) {
      if (fs::exists(in.path())) {
        LOG(warn) << "parsing " << in.path();
        if (config_.instant_forward_) {
          publisher pub;
          parse_sequential(in, pub);
        } else {
          parse_sequential(in, null_pub_);
        }
      } else {
        LOG(warn) << in.path() << " does not exist";
      }
    }

    if (config_.init_time_.unix_time_ != 0) {
      forward(config_.init_time_.unix_time_);
    }
  }

  static std::string_view get_content_type(HTTPRequest const* req) {
    for (auto const& h : *req->headers()) {
      if (std::string_view{h->name()->c_str(), h->name()->size()} ==
          "Content-Type") {
        return {h->value()->c_str(), h->value()->size()};
      }
    }
    return {};
  }

  msg_ptr upload(msg_ptr const& msg) {
    auto const req = motis_content(HTTPRequest, msg);
    auto const content = req->content();
    auto const ft =
        guess_file_type(get_content_type(req),
                        std::string_view{content->c_str(), content->size()});
    publisher pub;

    parse_str_and_write_to_db(file_upload_, {content->c_str(), content->size()},
                              ft, pub);

    sched_.system_time_ = pub.max_timestamp_;
    sched_.last_update_timestamp_ = std::time(nullptr);
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
    return {};
  }

  msg_ptr read(msg_ptr const&) {
    publisher pub;
    for (auto& in : inputs_) {
      parse_sequential(in, pub);
    }
    sched_.system_time_ = pub.max_timestamp_;
    sched_.last_update_timestamp_ = std::time(nullptr);
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
    return {};
  }

  msg_ptr forward(msg_ptr const& msg) {
    forward(motis_content(RISForwardTimeRequest, msg)->new_time());
    return {};
  }

  msg_ptr purge(msg_ptr const& msg) {
    auto const until =
        static_cast<unixtime>(motis_content(RISPurgeRequest, msg)->until());

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
          static_cast<unixtime>(
              flatbuffers::GetRoot<Message>(reinterpret_cast<void const*>(ptr))
                  ->timestamp()));
      offsets_.push_back(
          CreateMessageHolder(fbb_, fbb_.CreateVector(ptr, size)));
    }

    size_t size() const { return offsets_.size(); }

    message_creator fbb_;
    std::vector<flatbuffers::Offset<MessageHolder>> offsets_;
    unixtime max_timestamp_ = 0;
  };

  struct null_publisher {
    void flush() {}
    void add(uint8_t const*, size_t const) {}
    size_t size() const { return 0; }  // NOLINT
    unixtime max_timestamp_ = 0;
  } null_pub_;

  void forward(unixtime const to) {
    auto const first_schedule_event_day =
        sched_.first_event_schedule_time_ !=
                std::numeric_limits<unixtime>::max()
            ? floor(sched_.first_event_schedule_time_,
                    static_cast<unixtime>(SECONDS_A_DAY))
            : external_schedule_begin(sched_);
    auto const last_schedule_event_day =
        sched_.last_event_schedule_time_ != std::numeric_limits<unixtime>::min()
            ? ceil(sched_.last_event_schedule_time_,
                   static_cast<unixtime>(SECONDS_A_DAY))
            : external_schedule_end(sched_);
    auto const min_timestamp =
        get_min_timestamp(first_schedule_event_day, last_schedule_event_day);
    if (min_timestamp) {
      forward(std::max(*min_timestamp, sched_.system_time_ + 1), to);
    } else {
      LOG(info) << "ris database has no relevant data";
    }
  }

  void forward(unixtime const from, unixtime const to) {
    LOG(info) << "forwarding from " << logging::time(from) << " to "
              << logging::time(to);

    auto t = db::txn{env_, db::txn_flags::RDONLY};
    auto db = t.dbi_open(MSG_DB);
    auto c = db::cursor{t, db};
    auto bucket = c.get(db::cursor_op::SET_RANGE, from);
    auto batch_begin = bucket ? bucket->first : 0;
    publisher pub;
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
            msg->timestamp() <= to && msg->timestamp() >= from) {
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
    sched_.system_time_ = to;
    ctx::await_all(motis_publish(make_no_msg("/ris/system_time_changed")));
  }

  std::optional<unixtime> get_min_timestamp(unixtime const from_day,
                                            unixtime const to_day) {
    utl::verify(from_day % SECONDS_A_DAY == 0, "from not a day");
    utl::verify(to_day % SECONDS_A_DAY == 0, "to not a day");

    constexpr auto const max = std::numeric_limits<unixtime>::max();

    auto min = max;
    auto t = db::txn{env_, db::txn_flags::RDONLY};
    auto db = t.dbi_open(MIN_DAY_DB);
    for (auto d = from_day; d != to_day; d += SECONDS_A_DAY) {
      auto const r = t.get(db, d);
      if (r) {
        min = std::min(min, to_unixtime(*r));
      }
    }

    return min != max ? std::make_optional(min) : std::nullopt;
  }

  enum class file_type { NONE, ZST, ZIP, XML, PROTOBUF, JSON };

  static file_type get_file_type(fs::path const& p) {
    if (p.extension() == ".zst") {
      return file_type::ZST;
    } else if (p.extension() == ".zip") {
      return file_type::ZIP;
    } else if (p.extension() == ".xml") {
      return file_type::XML;
    } else if (p.extension() == ".pb") {
      return file_type::PROTOBUF;
    } else if (p.extension() == ".json") {
      return file_type::JSON;
    } else {
      return file_type::NONE;
    }
  }

  static file_type guess_file_type(std::string_view content_type,
                                   std::string_view content) {
    using boost::algorithm::iequals;
    using boost::algorithm::starts_with;

    if (iequals(content_type, "application/zip")) {
      return file_type::ZIP;
    } else if (iequals(content_type, "application/zstd")) {
      return file_type::ZST;
    } else if (iequals(content_type, "application/xml") ||
               iequals(content_type, "text/xml")) {
      return file_type::XML;
    } else if (iequals(content_type, "application/json") ||
               iequals(content_type, "application/x.ribasis")) {
      return file_type::JSON;
    }

    if (content.size() < 4) {
      return file_type::NONE;
    } else if (starts_with(content, "PK")) {
      return file_type::ZIP;
    } else if (starts_with(content, "\x28\xb5\x2f\xfd")) {
      return file_type::ZST;
    } else if (starts_with(content, "<")) {
      return file_type::XML;
    } else if (starts_with(content, "{")) {
      return file_type::JSON;
    }

    return file_type::NONE;
  }

  std::vector<std::tuple<unixtime, fs::path, file_type>> collect_files(
      fs::path const& p) {
    if (fs::is_regular_file(p)) {
      if (auto const t = get_file_type(p);
          t != file_type::NONE && !is_known_file(p)) {
        return {std::make_tuple(fs::last_write_time(p), p, t)};
      }
    } else if (fs::is_directory(p)) {
      std::vector<std::tuple<unixtime, fs::path, file_type>> files;
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
            parse_file_and_write_to_db(std::get<1>(e), std::get<2>(e), pub);
          });
        }));
    env_.force_sync();
  }

  template <typename Publisher>
  void parse_sequential(input& in, Publisher& pub) {
    for (auto const& [t, path, type] :
         collect_files(fs::canonical(in.path(), in.path().root_path()))) {
      (void)t;
      parse_file_and_write_to_db(in, path, type, pub);
      if (config_.instant_forward_) {
        sched_.system_time_ = pub.max_timestamp_;
        sched_.last_update_timestamp_ = std::time(nullptr);
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
  void parse_file_and_write_to_db(input& in, fs::path const& p,
                                  file_type const type, Publisher& pub) {
    using tar_zst = tar_reader<zstd_reader>;
    auto const& cp = p.generic_string();

    try {
      switch (type) {
        case file_type::ZST:
          parse_and_write_to_db(in, tar_zst(zstd_reader(cp.c_str())), type,
                                pub);
          break;
        case file_type::ZIP:
          parse_and_write_to_db(in, zip_reader(cp.c_str()), type, pub);
          break;
        case file_type::XML:
        case file_type::PROTOBUF:
        case file_type::JSON:
          parse_and_write_to_db(in, file_reader(cp.c_str()), type, pub);
          break;
        default: assert(false);
      }
    } catch (...) {
      LOG(logging::error) << "failed to read " << p;
    }
    add_to_known_files(p);
  }

  template <typename Publisher>
  void parse_str_and_write_to_db(input& in, std::string_view sv,
                                 file_type const type, Publisher& pub) {
    switch (type) {
      case file_type::ZST:
        throw utl::fail("zst upload is not supported");
        break;
      case file_type::ZIP:
        parse_and_write_to_db(in, zip_reader{sv.data(), sv.size()}, type, pub);
        break;
      case file_type::XML:
      case file_type::PROTOBUF:
      case file_type::JSON:
        parse_and_write_to_db(in, string_view_reader{sv}, type, pub);
        break;
      default: assert(false);
    }
  }

  template <typename Reader, typename Publisher>
  void parse_and_write_to_db(input& in, Reader&& reader, file_type const type,
                             Publisher& pub) {
    auto const risml_fn = [&](std::string_view s, std::string_view,
                              std::function<void(ris_message &&)> const& cb) {
      risml::to_ris_message(s, cb, in.tag());
    };
    auto const gtfsrt_fn = [&](std::string_view s, std::string_view,
                               std::function<void(ris_message &&)> const& cb) {
      gtfsrt::to_ris_message(sched_, in.gtfs_knowledge(),
                             config_.gtfs_is_addition_skip_allowed_, s, cb,
                             in.tag());
    };
    auto const ribasis_fn = [&](std::string_view s, std::string_view,
                                std::function<void(ris_message &&)> const& cb) {
      ribasis::to_ris_message(s, cb, in.tag());
    };
    auto const file_fn = [&](std::string_view s, std::string_view file_name,
                             std::function<void(ris_message &&)> const& cb) {
      if (boost::ends_with(file_name, ".xml")) {
        return risml_fn(s, file_name, cb);
      } else if (boost::ends_with(file_name, ".json")) {
        return ribasis_fn(s, file_name, cb);
      } else if (boost::ends_with(file_name, ".pb")) {
        return gtfsrt_fn(s, file_name, cb);
      }
    };

    switch (type) {
      case file_type::ZST:
      case file_type::ZIP: write_to_db(reader, file_fn, pub); break;
      case file_type::XML: write_to_db(reader, risml_fn, pub); break;
      case file_type::PROTOBUF: write_to_db(reader, gtfsrt_fn, pub); break;
      case file_type::JSON: write_to_db(reader, ribasis_fn, pub); break;
      default: assert(false);
    }
  }

  template <typename Reader, typename ParserFn, typename Publisher>
  void write_to_db(Reader&& reader, ParserFn parser_fn, Publisher& pub) {
    std::map<unixtime /* d.b */, unixtime /* min(t) : e <= d.e && l >= d.b */>
        min;
    std::map<unixtime /* d.b */, unixtime /* max(t) : e <= d.e && l >= d.b */>
        max;
    std::map<unixtime /* tout */, std::vector<char>> buf;
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

      for_each_day(m, [&](unixtime const d) {
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
      parse(*reader_content, reader.current_file_name(),
            [&](ris_message&& m) { write(std::move(m)); });
    }

    flush_to_db();
    update_min_max(min, max);
    pub.flush();
  }

  void update_min_max(std::map<unixtime, unixtime> const& min,
                      std::map<unixtime, unixtime> const& max) {
    std::lock_guard<std::mutex> lock{min_max_mutex_};

    auto t = db::txn{env_};
    auto min_db = t.dbi_open(MIN_DAY_DB);
    auto max_db = t.dbi_open(MAX_DAY_DB);

    for (auto const [day, min_timestamp] : min) {
      auto smallest = min_timestamp;
      if (auto entry = t.get(min_db, day); entry) {
        smallest = std::min(smallest, to_unixtime(*entry));
      }
      t.put(min_db, day, from_unixtime(smallest));
    }

    for (auto const [day, max_timestamp] : max) {
      auto largest = max_timestamp;
      if (auto entry = t.get(max_db, day); entry) {
        largest = std::max(largest, to_unixtime(*entry));
      }
      t.put(max_db, day, from_unixtime(largest));
    }

    t.commit();
  }

  schedule& sched_;
  db::env env_;
  std::atomic<uint64_t> next_msg_id_{0};
  std::mutex min_max_mutex_;
  std::mutex merge_mutex_;

  config const& config_;

  input file_upload_{""};
  std::vector<input> inputs_;
};

ris::ris() : module("RIS", "ris") {
  param(config_.db_path_, "db", "ris database path");
  param(config_.input_, "input",
        "input paths. expected format [tag:]path (tag MUST match the "
        "timetable)");
  param(config_.db_max_size_, "db_max_size", "virtual memory map size");
  param(config_.init_time_, "init_time", "initial forward time");
  param(config_.clear_db_, "clear_db", "clean db before init");
  param(config_.instant_forward_, "instant_forward",
        "automatically forward after every file during read");
  param(config_.gtfs_is_addition_skip_allowed_,
        "gtfsrt.is_addition_skip_allowed", "allow skips on additional trips");
}

ris::~ris() = default;

void ris::reg_subc(motis::module::subc_reg& r) {
  r.register_cmd(
      "gtfsrt-json2pb", "json to protobuf", [](int argc, char const** argv) {
        if (argc != 3) {
          std::cout << "usage: " << argv[0] << " JSON_FILE PB_OUTPUT\n";
          return 1;
        }

        auto const file = utl::read_file(argv[1]);
        if (!file.has_value()) {
          std::cout << "unable to read file " << argv[1] << "\n";
          return 1;
        }

        auto const out = gtfsrt::json_to_protobuf(*file);
        utl::file{argv[2], "w"}.write(&out[0], out.size());

        return 0;
      });
  r.register_cmd("gtfsrt-pb2json", "protobuf to json",
                 [](int argc, char const** argv) {
                   if (argc != 2) {
                     std::cout << "usage: " << argv[0] << " PB_FILE\n";
                     return 1;
                   }

                   auto const file = utl::read_file(argv[1]);
                   if (!file.has_value()) {
                     std::cout << "unable to read file " << argv[1] << "\n";
                     return 1;
                   }

                   std::cout << gtfsrt::protobuf_to_json(*file) << "\n";

                   return 0;
                 });
}

void ris::init(motis::module::registry& r) {
  impl_ =
      std::make_unique<impl>(*const_cast<schedule*>(&get_sched()),  // NOLINT
                             config_);
  r.subscribe(
      "/init",
      [this]() {
        impl_->init();  // NOLINT
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/upload", [this](auto&& m) { return impl_->upload(m); },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/forward", [this](auto&& m) { return impl_->forward(m); },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/read", [this](auto&& m) { return impl_->read(m); },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/purge", [this](auto&& m) { return impl_->purge(m); },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::SCHEDULE),
          ctx::access_t::WRITE}});
}

}  // namespace motis::ris
