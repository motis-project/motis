#include "motis/ris/ris.h"

#include <chrono>
#include <cstdint>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>

#include "boost/algorithm/string/predicate.hpp"

#include "utl/concat.h"
#include "utl/parser/file.h"
#include "utl/read_file.h"

#include "net/http/client/url.h"

#include "conf/date_time.h"

#include "tar/file_reader.h"
#include "tar/tar_reader.h"
#include "tar/zstd_reader.h"

#include "utl/overloaded.h"
#include "utl/verify.h"
#include "utl/zip.h"

#include "lmdb/lmdb.hpp"

#include "rabbitmq/amqp.hpp"

#include "motis/core/common/logging.h"
#include "motis/core/common/unixtime.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/print_trip.h"

#include "motis/module/context/motis_http_req.h"
#include "motis/module/context/motis_publish.h"
#include "motis/module/context/motis_spawn.h"
#include "motis/module/event_collector.h"

#include "motis/ris/amqp_buffer_reader.h"
#include "motis/ris/ris_message.h"
#include "motis/ris/string_view_reader.h"
#include "motis/ris/zip_reader.h"

#include "motis/ris/risml/risml_parser.h"

#include "motis/ris/gtfs-rt/common.h"
#include "motis/ris/gtfs-rt/gtfsrt_parser.h"
#include "motis/ris/gtfs-rt/util.h"

#include "motis/ris/ribasis/ribasis_parser.h"
#include "motis/ris/ribasis/ribasis_receiver.h"

namespace fs = std::filesystem;
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

// stores the latest stream offset received for each rabbitmq stream
// key: rabbitmq stream identifier
// value: stream offset of the latest message that was received
constexpr auto const STREAM_OFFSET_DB = "STREAM_OFFSET_DB";

// stores the timestamp of the last message received for each rabbitmq stream
// key: rabbitmq stream identifier
// value: timestamp of the latest message that was received
constexpr auto const STREAM_TIMESTAMP_DB = "STREAM_TIMESTAMP_DB";

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

inline unixtime to_unixtime(std::string_view s) {
  return *reinterpret_cast<unixtime const*>(s.data());
}

inline unixtime unixtime_duration_ago(conf::duration const& duration) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             (std::chrono::system_clock::now() - duration.duration_)
                 .time_since_epoch())
      .count();
}

std::string_view from_unixtime(unixtime const& t) {
  return {reinterpret_cast<char const*>(&t), sizeof(t)};
}

template <typename Publisher>
inline void update_system_time(schedule& sched, Publisher const& pub) {
  sched.system_time_ = std::max(sched.system_time_, pub.max_timestamp_);
  sched.last_update_timestamp_ = std::time(nullptr);
}

template <typename Fn>
inline void for_each_rabbitmq_config(config& ris_config, Fn&& fn) {
  for (auto& [config, prefix] : {std::pair<rabbitmq_config&, std::string>{
                                     ris_config.rabbitmq1_, "rabbitmq1"},
                                 std::pair<rabbitmq_config&, std::string>{
                                     ris_config.rabbitmq2_, "rabbitmq2"}}) {
    fn(config, prefix);
  }
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
    using source_t = std::variant<net::http::client::request, fs::path>;
    enum class source_type { path, url };

    input(schedule const& sched, config const& c, std::string const& in)
        : input{sched, c, split(in)} {}

    std::string str() const {
      return std::visit(
          utl::overloaded{
              [](fs::path const& p) { return "path: " + p.generic_string(); },
              [](net::http::client::request const& u) {
                auto const auth_it = u.headers.find("Authorization");
                return "url: " + u.address.str() + ", auth: " +
                       (auth_it == end(u.headers) ? "none" : auth_it->second);
              }},
          src_);
    }

    source_type source_type() const {
      return std::visit(
          utl::overloaded{[](fs::path const&) { return source_type::path; },
                          [](net::http::client::request const&) {
                            return source_type::url;
                          }},
          src_);
    }

    fs::path get_path() const {
      utl::verify(std::holds_alternative<fs::path>(src_), "no path {}", str());
      return std::get<fs::path>(src_);
    }

    net::http::client::request get_request() const {
      utl::verify(std::holds_alternative<net::http::client::request>(src_),
                  "no url {}", str());
      auto req = std::get<net::http::client::request>(src_);
      return config_.http_proxy_.empty() ? req
                                         : req.set_proxy(config_.http_proxy_);
    }

    std::string const& tag() const { return tag_; }

    gtfsrt::knowledge_context& gtfs_knowledge() { return gtfs_knowledge_; }

  private:
    input(schedule const& sched, config const& c,
          std::pair<source_t, std::string>&& path_and_tag)
        : config_{c},
          src_{std::move(path_and_tag.first)},
          tag_{path_and_tag.second},
          gtfs_knowledge_{path_and_tag.second, sched} {}

    static std::pair<source_t, std::string> split(std::string const& in) {
      auto const is_url = [](std::string const& s) {
        return boost::starts_with(s, "http://") ||
               boost::starts_with(s, "https://");
      };

      auto const parse_req = [](std::string const& s) {
        if (auto const delimiter_pos = s.find('|');
            delimiter_pos != std::string::npos) {
          auto req = net::http::client::request{s.substr(0, delimiter_pos)};
          req.headers.emplace("Authorization", s.substr(delimiter_pos + 1));
          return req;
        } else {
          return net::http::client::request{s};
        }
      };

      std::string tag;
      if (auto const delimiter_pos = in.find('|');
          delimiter_pos != std::string::npos) {
        tag = in.substr(0, delimiter_pos);
        tag = tag.empty() ? "" : tag + "_";
        auto const src = in.substr(delimiter_pos + 1);
        return {
            is_url(src) ? source_t{parse_req(src)} : source_t{fs::path{src}},
            tag};
      } else {
        return {is_url(in) ? source_t{parse_req(in)} : source_t{fs::path{in}},
                tag};
      }
    }

    config const& config_;

    source_t src_;
    std::string tag_;

    gtfsrt::knowledge_context gtfs_knowledge_;
    std::unique_ptr<amqp::ssl_connection> con_;
  };

  explicit impl(config& c) : config_{c} {}

  void init_ribasis_receivers(dispatcher* d, schedule* sched) {
    for_each_rabbitmq_config(config_, [this, &d, &sched](
                                          rabbitmq_config& config,
                                          std::string const& prefix) {
      utl::verify(config.login_.valid(), "invalid {} configuration", prefix);
      if (config.login_.empty()) {
        return;
      }

      if (config.resume_stream_) {
        auto const queue_id = ribasis::get_queue_id(config.login_);
        auto const stored_stream_offset = get_stream_offset(queue_id);
        auto const stored_stream_timestamp = get_stream_timestamp(queue_id);
        if (stored_stream_offset) {
          auto resume = true;
          if (config.max_resume_age_ && stored_stream_timestamp) {
            if (*stored_stream_timestamp <
                unixtime_duration_ago(config.max_resume_age_)) {
              LOG(info) << prefix
                        << ": last stream timestamp is too old, resuming at "
                        << config.login_.stream_offset_;
              resume = false;
            }
          }
          if (resume) {
            LOG(info) << prefix << ": resuming at stored stream offset "
                      << *stored_stream_offset;
            config.login_.numeric_stream_offset_ = *stored_stream_offset + 1;
          }
        } else {
          LOG(info) << prefix << ": no stored stream offset found, resuming at "
                    << config.login_.stream_offset_;
        }
      }

      ribasis_receivers_.emplace_back(std::make_unique<ribasis::receiver>(
          config, [this, d, sched](ribasis::receiver& rec,
                                   std::vector<amqp::msg>&& msgs) {
            d->enqueue(
                ctx_data{d},
                [this, &rec, sched, msgs = std::move(msgs)]() {
                  publisher pub;
                  pub.schedule_res_id_ =
                      to_res_id(::motis::module::global_res_id::SCHEDULE);

                  LOG(info) << rec.name() << ": processing " << msgs.size()
                            << " messages";

                  parse_and_write_to_db(*file_upload_, amqp_buffer_reader{msgs},
                                        file_type::JSON, pub);

                  auto const stream_offset = msgs.back().stream_offset_;
                  if (stream_offset) {
                    LOG(info) << rec.name()
                              << ": new stream offset: " << *stream_offset
                              << ", queue id: " << rec.queue_id();
                    store_stream_offset(rec.queue_id(), *stream_offset);
                  }

                  if (pub.max_timestamp_ != 0) {
                    store_stream_timestamp(rec.queue_id(), pub.max_timestamp_);
                  }

                  LOG(info) << rec.name() << ": system time: old="
                            << format_unix_time(sched->system_time_)
                            << ", max=" << format_unix_time(pub.max_timestamp_);

                  update_system_time(*sched, pub);

                  publish_system_time_changed(pub.schedule_res_id_);
                },
                ctx::op_id{"ribasis_receive_" + rec.name(), CTX_LOCATION, 0U},
                ctx::op_type_t::IO,
                ctx::accesses_t{ctx::access_request{
                    to_res_id(::motis::module::global_res_id::RIS_DATA),
                    ctx::access_t::WRITE}});
          }));
    });
  }

  void update_gtfs_rt(schedule& sched) {
    auto futures = std::vector<http_future_t>{};
    auto inputs = std::vector<input*>{};
    for (auto& in : inputs_) {
      if (in.source_type() == input::source_type::url) {
        futures.emplace_back(motis_http(in.get_request()));
        inputs.emplace_back(&in);
      }
    }

    publisher pub;
    pub.schedule_res_id_ = to_res_id(::motis::module::global_res_id::SCHEDULE);

    for (auto const& [f, in] : utl::zip(futures, inputs)) {
      try {
        parse_str_and_write_to_db(*in, f->val().body, file_type::PROTOBUF, pub);
      } catch (std::exception const& e) {
        LOG(logging::error)
            << "input source \"" << in->str() << "\": " << e.what();
      }
    }

    update_system_time(sched, pub);
    publish_system_time_changed(pub.schedule_res_id_);
  }

  void init(dispatcher& d, schedule& sched) {
    inputs_ = utl::to_vec(config_.input_, [&](std::string const& in) {
      return input{sched, config_, in};
    });
    file_upload_ = std::make_unique<input>(sched, config_, "");

    if (config_.clear_db_ && fs::exists(config_.db_path_)) {
      LOG(info) << "clearing database path " << config_.db_path_;
      std::error_code ec;
      fs::remove_all(config_.db_path_, ec);
      fs::remove_all(config_.db_path_ + "-lock", ec);
    }

    env_.set_maxdbs(6);
    env_.set_mapsize(config_.db_max_size_);

    try {
      env_.open(config_.db_path_.c_str(),
                lmdb::env_open_flags::NOSUBDIR | lmdb::env_open_flags::NOTLS);
    } catch (...) {
      l(logging::error, "ris: can't open database {}", config_.db_path_);
      throw;
    }

    db::txn t{env_};
    t.dbi_open(FILE_DB, db::dbi_flags::CREATE);
    t.dbi_open(MSG_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MIN_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(MAX_DAY_DB, db::dbi_flags::CREATE | db::dbi_flags::INTEGERKEY);
    t.dbi_open(STREAM_OFFSET_DB, db::dbi_flags::CREATE);
    t.dbi_open(STREAM_TIMESTAMP_DB, db::dbi_flags::CREATE);
    t.commit();

    if (config_.init_purge_) {
      LOG(info) << "removing ris messages older than " << config_.init_purge_
                << " from database";
      auto const deleted = purge(unixtime_duration_ago(config_.init_purge_));
      LOG(info) << "removed " << deleted << " old messages";
    }

    std::vector<input> const urls;
    for (auto& in : inputs_) {
      if (in.source_type() != input::source_type::path) {
        continue;
      }

      if (fs::exists(in.get_path())) {
        LOG(warn) << "parsing " << in.get_path();
        if (config_.instant_forward_) {
          publisher pub;
          parse_sequential(sched, in, pub);
        } else {
          parse_sequential(sched, in, null_pub_);
        }
      } else {
        LOG(warn) << in.get_path() << " does not exist";
      }
    }

    auto const has_urls = std::any_of(
        begin(inputs_), end(inputs_),
        [](auto&& in) { return in.source_type() == input::source_type::url; });
    if (has_urls) {
      d.register_timer(
          "RIS GTFS-RT Update",
          boost::posix_time::seconds{config_.gtfs_rt_update_interval_},
          [this, &sched]() { update_gtfs_rt(sched); },
          ctx::accesses_t{ctx::access_request{
              to_res_id(::motis::module::global_res_id::RIS_DATA),
              ctx::access_t::WRITE}});
    }

    if (config_.init_time_.unix_time_ != 0) {
      forward(sched, 0U, config_.init_time_.unix_time_, true);
    }

    init_ribasis_receivers(&d, &sched);
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

  msg_ptr upload(schedule& sched, msg_ptr const& msg) {
    auto const req = motis_content(HTTPRequest, msg);
    auto const content = req->content();
    auto const ft =
        guess_file_type(get_content_type(req),
                        std::string_view{content->c_str(), content->size()});
    publisher pub;

    parse_str_and_write_to_db(*file_upload_,
                              {content->c_str(), content->size()}, ft, pub);

    update_system_time(sched, pub);
    publish_system_time_changed(pub.schedule_res_id_);
    return {};
  }

  msg_ptr read(schedule& sched, msg_ptr const&) {
    publisher pub;
    for (auto& in : inputs_) {
      if (in.source_type() == input::source_type::path) {
        parse_sequential(sched, in, pub);
      }
    }
    update_system_time(sched, pub);
    publish_system_time_changed(pub.schedule_res_id_);
    return {};
  }

  msg_ptr forward(module& mod, msg_ptr const& msg) {
    auto const req = motis_content(RISForwardTimeRequest, msg);
    auto const schedule_res_id =
        req->schedule() == 0U ? to_res_id(global_res_id::SCHEDULE)
                              : static_cast<ctx::res_id_t>(req->schedule());
    auto res_lock =
        mod.lock_resources({{schedule_res_id, ctx::access_t::WRITE}});
    auto& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;
    forward(sched, schedule_res_id,
            motis_content(RISForwardTimeRequest, msg)->new_time());
    return {};
  }

  msg_ptr purge(msg_ptr const& msg) {
    auto const until =
        static_cast<unixtime>(motis_content(RISPurgeRequest, msg)->until());

    purge(until);

    return {};
  }

  std::uint64_t purge(unixtime const until) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(MSG_DB);
    auto c = db::cursor{t, db};
    auto bucket = c.get(db::cursor_op::SET_RANGE, until);
    if (!bucket) {
      // no entry >= until found, maybe all messages are older
      bucket = c.get(db::cursor_op::LAST, 0);
    }
    auto deleted = 0ULL;
    while (bucket) {
      if (bucket->first <= until) {
        c.del();
        ++deleted;
      }
      bucket = c.get(db::cursor_op::PREV, 0);
    }
    t.commit();
    c.reset();
    return deleted;
  }

  msg_ptr apply(module& mod, msg_ptr const& msg) {
    auto const req = motis_content(RISApplyRequest, msg);
    auto const schedule_res_id =
        req->schedule() == 0U ? to_res_id(global_res_id::SCHEDULE)
                              : static_cast<ctx::res_id_t>(req->schedule());
    auto res_lock =
        mod.lock_resources({{schedule_res_id, ctx::access_t::WRITE}});
    auto& sched = *res_lock.get<schedule_data>(schedule_res_id).schedule_;

    publisher pub{schedule_res_id};
    auto successful = 0ULL;
    auto failed = 0ULL;
    for (auto const& rim : *req->input_messages()) {
      if (parse_and_publish_message(rim, pub)) {
        ++successful;
      } else {
        ++failed;
      }
    }

    pub.flush();
    update_system_time(sched, pub);

    publish_system_time_changed(schedule_res_id);

    message_creator mc;
    mc.create_and_finish(
        MsgContent_RISApplyResponse,
        CreateRISApplyResponse(mc, sched.system_time_, successful, failed)
            .Union());
    return make_msg(mc);
  }

  struct publisher {
    publisher() = default;
    explicit publisher(ctx::res_id_t schedule_res_id)
        : schedule_res_id_{schedule_res_id} {}
    publisher(publisher&&) = delete;
    publisher(publisher const&) = delete;
    publisher& operator=(publisher&&) = delete;
    publisher& operator=(publisher const&) = delete;

    ~publisher() { flush(); }

    void flush() {
      if (offsets_.empty() || skip_flush_) {
        return;
      }
      // prevent the destructor from running flush() again if an exception
      // occurs inside flush()
      skip_flush_ = true;

      fbb_.create_and_finish(
          MsgContent_RISBatch,
          CreateRISBatch(fbb_, fbb_.CreateVector(offsets_), schedule_res_id_)
              .Union(),
          "/ris/messages");

      auto msg = make_msg(fbb_);
      fbb_.Clear();
      offsets_.clear();

      ctx::await_all(motis_publish(msg));
      skip_flush_ = false;
    }

    void add(uint8_t const* ptr, size_t const size) {
      max_timestamp_ =
          std::max(max_timestamp_,
                   static_cast<unixtime>(flatbuffers::GetRoot<RISMessage>(
                                             reinterpret_cast<void const*>(ptr))
                                             ->timestamp()));
      offsets_.push_back(
          CreateRISMessageHolder(fbb_, fbb_.CreateVector(ptr, size)));
    }

    size_t size() const { return offsets_.size(); }

    message_creator fbb_;
    std::vector<flatbuffers::Offset<RISMessageHolder>> offsets_;
    unixtime max_timestamp_ = 0;
    bool skip_flush_{false};
    ctx::res_id_t schedule_res_id_{0U};
  };

  struct null_publisher {
    void flush() {}
    void add(uint8_t const*, size_t const) {}
    size_t size() const { return 0; }  // NOLINT
    unixtime max_timestamp_ = 0;
    ctx::res_id_t schedule_res_id_{};
  } null_pub_;

  void forward(schedule& sched, ctx::res_id_t schedule_res_id,
               unixtime const to, bool force_update_system_time = false) {
    auto const first_schedule_event_day =
        sched.first_event_schedule_time_ != std::numeric_limits<unixtime>::max()
            ? floor(sched.first_event_schedule_time_,
                    static_cast<unixtime>(SECONDS_A_DAY))
            : external_schedule_begin(sched);
    auto const last_schedule_event_day =
        sched.last_event_schedule_time_ != std::numeric_limits<unixtime>::min()
            ? ceil(sched.last_event_schedule_time_,
                   static_cast<unixtime>(SECONDS_A_DAY))
            : external_schedule_end(sched);
    auto const min_timestamp =
        get_min_timestamp(first_schedule_event_day, last_schedule_event_day);
    if (min_timestamp) {
      forward(sched, schedule_res_id,
              std::max(*min_timestamp, sched.system_time_ + 1), to);
    } else {
      LOG(info) << "ris database has no relevant data";
      if (force_update_system_time) {
        LOG(info) << "updating system time to " << format_unix_time(to)
                  << " anyway";
        sched.system_time_ = to;
      }
    }
  }

  void forward(schedule& sched, ctx::res_id_t schedule_res_id,
               unixtime const from, unixtime const to) {
    LOG(info) << "forwarding from " << logging::time(from) << " to "
              << logging::time(to) << " [schedule " << schedule_res_id << "]";

    auto t = db::txn{env_, db::txn_flags::RDONLY};
    auto db = t.dbi_open(MSG_DB);
    auto c = db::cursor{t, db};
    auto bucket = c.get(db::cursor_op::SET_RANGE, from);
    auto batch_begin = bucket ? bucket->first : 0;
    publisher pub{schedule_res_id};
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

        if (auto const msg = GetRISMessage(ptr);
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
    sched.system_time_ = to;
    publish_system_time_changed(pub.schedule_res_id_);
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
        return {std::make_tuple(
            static_cast<unixtime>(
                std::chrono::time_point_cast<std::chrono::seconds>(
                    fs::last_write_time(p))
                    .time_since_epoch()
                    .count()),
            p, t)};
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
  void parse_sequential(schedule& sched, input& in, Publisher& pub) {
    if (!fs::exists(in.get_path())) {
      l(logging::error, "ris input path {} does not exist", in.get_path());
      return;
    }

    for (auto const& [t, path, type] :
         collect_files(fs::canonical(in.get_path()))) {
      try {
        parse_file_and_write_to_db(in, path, type, pub);
        if (config_.instant_forward_) {
          update_system_time(sched, pub);
          try {
            publish_system_time_changed(pub.schedule_res_id_);
          } catch (std::system_error& e) {
            LOG(info) << e.what();
          }
        }
      } catch (std::exception const& e) {
        l(logging::error, "error parsing file {}", path);
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
                              std::function<void(ris_message&&)> const& cb) {
      risml::to_ris_message(s, cb, in.tag());
    };
    auto const gtfsrt_fn = [&](std::string_view s, std::string_view,
                               std::function<void(ris_message&&)> const& cb) {
      gtfsrt::to_ris_message(in.gtfs_knowledge(),
                             config_.gtfs_is_addition_skip_allowed_, s, cb,
                             in.tag());
    };
    auto const ribasis_fn = [&](std::string_view s, std::string_view,
                                std::function<void(ris_message&&)> const& cb) {
      ribasis::to_ris_message(s, cb, in.tag());
    };
    auto const file_fn = [&](std::string_view s, std::string_view file_name,
                             std::function<void(ris_message&&)> const& cb) {
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

      std::lock_guard<std::mutex> const lock{merge_mutex_};

      auto t = db::txn{env_};
      auto db = t.dbi_open(MSG_DB);
      auto c = db::cursor{t, db};

      for (auto& [timestamp, entry] : buf) {
        if (auto const v = c.get(lmdb::cursor_op::SET_RANGE, timestamp);
            v && v->first == timestamp) {
          entry.insert(end(entry), begin(v->second), end(v->second));
        }
        c.put(timestamp, std::string_view{entry.data(), entry.size()});
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
      std::memcpy(buf_val.data() + base, &msg_size, SIZE_TYPE_SIZE);
      std::memcpy(buf_val.data() + base + SIZE_TYPE_SIZE, m.data(), m.size());

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
    std::lock_guard<std::mutex> const lock{min_max_mutex_};

    auto t = db::txn{env_};
    auto min_db = t.dbi_open(MIN_DAY_DB);
    auto max_db = t.dbi_open(MAX_DAY_DB);

    for (auto const& [day, min_timestamp] : min) {
      auto smallest = min_timestamp;
      if (auto entry = t.get(min_db, day); entry) {
        smallest = std::min(smallest, to_unixtime(*entry));
      }
      t.put(min_db, day, from_unixtime(smallest));
    }

    for (auto const& [day, max_timestamp] : max) {
      auto largest = max_timestamp;
      if (auto entry = t.get(max_db, day); entry) {
        largest = std::max(largest, to_unixtime(*entry));
      }
      t.put(max_db, day, from_unixtime(largest));
    }

    t.commit();
  }

  void store_stream_offset(std::string const& queue_id,
                           std::int64_t stream_offset) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(STREAM_OFFSET_DB);
    t.put(db, queue_id,
          std::string_view{reinterpret_cast<char const*>(&stream_offset),
                           sizeof(stream_offset)});
    t.commit();
  }

  std::optional<std::int64_t> get_stream_offset(std::string const& queue_id) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(STREAM_OFFSET_DB);
    if (auto const val = t.get(db, queue_id); val) {
      return {lmdb::as_int<std::int64_t>(*val)};
    } else {
      return {};
    }
  }

  void store_stream_timestamp(std::string const& queue_id, unixtime const ts) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(STREAM_TIMESTAMP_DB);
    t.put(db, queue_id,
          std::string_view{reinterpret_cast<char const*>(&ts), sizeof(ts)});
    t.commit();
  }

  std::optional<unixtime> get_stream_timestamp(std::string const& queue_id) {
    auto t = db::txn{env_};
    auto db = t.dbi_open(STREAM_TIMESTAMP_DB);
    if (auto const val = t.get(db, queue_id); val) {
      return {lmdb::as_int<unixtime>(*val)};
    } else {
      return {};
    }
  }

  static void publish_system_time_changed(ctx::res_id_t schedule_res_id) {
    message_creator mc;
    mc.create_and_finish(
        MsgContent_RISSystemTimeChanged,
        CreateRISSystemTimeChanged(mc, schedule_res_id).Union(),
        "/ris/system_time_changed");
    ctx::await_all(motis_publish(make_msg(mc)));
  }

  template <typename Publisher>
  bool parse_and_publish_message(RISInputMessage const* rim, Publisher& pub) {
    auto content_sv =
        std::string_view{rim->content()->c_str(), rim->content()->size()};

    auto const handle_message = [&](ris_message&& m) {
      pub.add(m.data(), m.size());
    };

    switch (rim->type()) {
      case RISContentType_RIBasis: {
        return ribasis::to_ris_message(content_sv, handle_message);
      }
      case RISContentType_RISML: {
        return risml::to_ris_message(content_sv, handle_message);
      }
      default: throw utl::fail("ris: unsupported message type");
    }
  }

  void stop_io() const {
    for (auto& rec : ribasis_receivers_) {
      if (rec) {
        rec->stop();
      }
    }
  }

  db::env env_;
  std::mutex min_max_mutex_;
  std::mutex merge_mutex_;

  config& config_;

  std::unique_ptr<input> file_upload_;
  std::vector<input> inputs_;

  std::vector<std::unique_ptr<ribasis::receiver>> ribasis_receivers_;
};

ris::ris() : module("RIS", "ris") {
  param(config_.db_path_, "db", "ris database path");
  param(config_.input_, "input",
        "input paths. expected format [tag:]path (tag MUST match the "
        "timetable)");
  param(config_.db_max_size_, "db_max_size", "virtual memory map size");
  param(config_.init_time_, "init_time", "initial forward time");
  param(config_.clear_db_, "clear_db", "clean db before init");
  param(config_.init_purge_, "init_purge",
        "remove messages older than this value from the database during "
        "startup (e.g. 12h, 1D)");
  param(config_.instant_forward_, "instant_forward",
        "automatically forward after every file during read");
  param(config_.gtfs_is_addition_skip_allowed_,
        "gtfsrt.is_addition_skip_allowed", "allow skips on additional trips");
  param(config_.http_proxy_, "http_proxy", "proxy for HTTP requests");
  param(config_.gtfs_rt_update_interval_, "gtfsrt.update_interval",
        "RT update interval in seconds for GTFS-RT");

  for_each_rabbitmq_config(
      config_, [this](rabbitmq_config& config, std::string const& prefix) {
        config.name_ = prefix;
        param(config.name_, prefix + ".name", "Name for logs");
        param(config.login_.host_, prefix + ".host", "RabbitMQ remote host");
        param(config.login_.port_, prefix + ".port", "RabbitMQ remote port");
        param(config.login_.user_, prefix + ".username", "RabbitMQ username");
        param(config.login_.pw_, prefix + ".password", "RabbitMQ password");
        param(config.login_.vhost_, prefix + ".vhost", "RabbitMQ vhost");
        param(config.login_.queue_, prefix + ".queue", "RabbitMQ queue name");
        param(config.login_.ca_, prefix + ".ca", "RabbitMQ path to CA file");
        param(config.login_.cert_, prefix + ".cert",
              "RabbitMQ path to client certificate");
        param(config.login_.key_, prefix + ".key",
              "RabbitMQ path to client key file");
        param(config.login_.prefetch_count_, prefix + ".prefetch_count",
              "Number of RabbitMQ messages to prefetch (must be > 0 if using "
              "RabbitMQ streams)");
        param(config.login_.stream_offset_, prefix + ".stream_offset",
              "Stream offset if using RabbitMQ streams (e.g. next, 2h, 1D...)");
        param(config.resume_stream_, prefix + ".resume_stream",
              "Resume stream at last received stream offset (if using RabbitMQ "
              "streams)");
        param(config.max_resume_age_, prefix + ".max_resume_age",
              "Only resume stream if last message is not older than this "
              "duration (e.g. 12h, 1D...), otherwise use stream_offset - "
              "empty/0 to always resume");
        param(config.update_interval_, prefix + ".update_interval",
              "RT update interval in seconds (RabbitMQ messages get buffered)");
      });
}

ris::~ris() = default;

void ris::stop_io() {
  if (impl_) {
    impl_->stop_io();
  }
}

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
        utl::file{argv[2], "w"}.write(out.data(), out.size());

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
  impl_ = std::make_unique<impl>(config_);
  add_shared_data(to_res_id(global_res_id::RIS_DATA), 0);
  r.subscribe(
      "/init",
      [this]() {
        impl_->init(*shared_data_,
                    const_cast<schedule&>(get_sched()));  // NOLINT
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::RIS_DATA),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/upload",
      [this](auto&& m) {
        return impl_->upload(const_cast<schedule&>(get_sched()), m);  // NOLINT
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::RIS_DATA),
          ctx::access_t::WRITE}});
  r.register_op("/ris/forward",
                [this](auto&& m) { return impl_->forward(*this, m); }, {});
  r.register_op(
      "/ris/read",
      [this](auto&& m) {
        return impl_->read(const_cast<schedule&>(get_sched()), m);  // NOLINT
      },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::RIS_DATA),
          ctx::access_t::WRITE}});
  r.register_op(
      "/ris/purge", [this](auto&& m) { return impl_->purge(m); },
      ctx::accesses_t{ctx::access_request{
          to_res_id(::motis::module::global_res_id::RIS_DATA),
          ctx::access_t::WRITE}});
  r.register_op("/ris/apply",
                [this](auto&& m) { return impl_->apply(*this, m); }, {});
}

void ris::import(motis::module::import_dispatcher& reg) {
  std::make_shared<motis::module::event_collector>(
      get_data_directory().generic_string(), "ris", reg,
      [this](motis::module::event_collector::dependencies_map_t const&,
             motis::module::event_collector::publish_fn_t const&) {
        import_successful_ = true;
      })
      ->require("SCHEDULE", [](motis::module::msg_ptr const& msg) {
        return msg->get()->content_type() == MsgContent_ScheduleEvent;
      });
}

bool ris::import_successful() const { return import_successful_; }

}  // namespace motis::ris
