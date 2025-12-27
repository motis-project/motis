#include "motis/tag_lookup.h"

#include <ctime>

#include "fmt/chrono.h"
#include "fmt/core.h"

#include "cista/io.h"

#include "utl/enumerate.h"
#include "utl/parser/split.h"
#include "utl/verify.h"

#include "net/bad_request_exception.h"
#include "net/not_found_exception.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/timetable.h"

namespace n = nigiri;

namespace motis {

trip_id<std::string_view> split_trip_id(std::string_view id) {
  auto const [date, start_time, tag, trip_id] =
      utl::split<'_', utl::cstr, utl::cstr, utl::cstr, utl::cstr>(id);

  auto ret = motis::trip_id{};

  utl::verify<net::bad_request_exception>(date.valid(),
                                          "invalid tripId date {}", id);
  ret.start_date_ = date.view();

  utl::verify<net::bad_request_exception>(start_time.valid(),
                                          "invalid tripId start_time {}", id);
  ret.start_time_ = start_time.view();

  utl::verify<net::bad_request_exception>(tag.valid(), "invalid tripId tag {}",
                                          id);
  ret.tag_ = tag.view();

  // allow trip ids starting with underscore
  auto const trip_id_len_plus_one =
      static_cast<std::size_t>(id.data() + id.size() - tag.str) - tag.length();
  utl::verify<net::bad_request_exception>(trip_id_len_plus_one > 1,
                                          "invalid tripId id {}", id);
  ret.trip_id_ =
      std::string_view{tag.str + tag.length() + 1, trip_id_len_plus_one - 1};

  return ret;
}

std::pair<std::string_view, std::string_view> split_tag_id(std::string_view x) {
  auto const first_underscore_pos = x.find('_');
  return first_underscore_pos != std::string_view::npos
             ? std::pair{x.substr(0, first_underscore_pos),
                         x.substr(first_underscore_pos + 1U)}
             : std::pair{std::string_view{}, x};
}

void tag_lookup::add(n::source_idx_t const src, std::string_view str) {
  utl::verify<net::bad_request_exception>(tag_to_src_.size() == to_idx(src),
                                          "invalid tag");
  tag_to_src_.emplace(std::string{str}, src);
  src_to_tag_.emplace_back(str);
}

n::source_idx_t tag_lookup::get_src(std::string_view tag) const {
  auto const it = tag_to_src_.find(tag);
  return it == end(tag_to_src_) ? n::source_idx_t::invalid() : it->second;
}

std::string_view tag_lookup::get_tag(n::source_idx_t const src) const {
  return src == n::source_idx_t::invalid() ? "" : src_to_tag_.at(src).view();
}

std::string tag_lookup::id(nigiri::timetable const& tt,
                           nigiri::location_idx_t const l) const {
  auto const src = tt.locations_.src_.at(l);
  auto const id = tt.locations_.ids_.at(l).view();
  return src == n::source_idx_t::invalid()
             ? std::string{id}
             : fmt::format("{}_{}", get_tag(src), id);
}

trip_id<std::string> tag_lookup::id_fragments(
    nigiri::timetable const& tt,
    n::rt::run_stop s,
    n::event_type const ev_type) const {
  if (s.fr_->is_scheduled()) {
    // trip id
    auto const t = s.get_trip_idx(ev_type);
    auto const id_idx = tt.trip_ids_[t].front();
    auto const id = tt.trip_id_strings_[id_idx].view();
    auto const src = tt.trip_id_src_[id_idx];

    // start date + start time
    auto const [day, gtfs_static_dep] = s.get_trip_start(ev_type);
    auto const start_hours = gtfs_static_dep / 60;
    auto const start_minutes = gtfs_static_dep % 60;

    return {
        fmt::format("{:%Y%m%d}", day),
        fmt::format("{:02}:{:02}", start_hours.count(), start_minutes.count()),
        std::string{get_tag(src)}, std::string{id}};
  } else {
    auto const id = s.fr_->id();
    auto const time = std::chrono::system_clock::to_time_t(
        (*s.fr_)[0].time(n::event_type::kDep));
    auto const utc = *std::gmtime(&time);
    auto const id_tag = get_tag(id.src_);
    auto const id_id = id.id_;
    return {fmt::format("{:04}{:02}{:02}", utc.tm_year + 1900, utc.tm_mon + 1,
                        utc.tm_mday),
            fmt::format("{:02}:{:02}", utc.tm_hour, utc.tm_min),
            std::string{id_tag}, std::string{id_id}};
  }
}

std::string tag_lookup::id(nigiri::timetable const& tt,
                           n::rt::run_stop s,
                           n::event_type const ev_type) const {
  auto const t = id_fragments(tt, s, ev_type);
  return fmt::format("{}_{}_{}_{}", std::move(t.start_date_),
                     std::move(t.start_time_), std::move(t.tag_),
                     std::move(t.trip_id_));
}

std::pair<nigiri::rt::run, nigiri::trip_idx_t> tag_lookup::get_trip(
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    std::string_view id) const {
  auto const split = split_trip_id(id);
  auto td = transit_realtime::TripDescriptor{};
  td.set_start_date(split.start_date_);
  td.set_start_time(split.start_time_);
  td.set_trip_id(split.trip_id_);
  return n::rt::gtfsrt_resolve_run({}, tt, rtt, get_src(split.tag_), td);
}

nigiri::location_idx_t tag_lookup::get_location(nigiri::timetable const& tt,
                                                std::string_view s) const {
  auto const [tag, id] = split_tag_id(s);
  auto const src = get_src(tag);
  try {
    return tt.locations_.location_id_to_idx_.at({{id}, src});
  } catch (...) {
    throw utl::fail<net::not_found_exception>(
        R"(Could not find timetable location "{}", tag="{}", id="{}", src={})",
        s, tag, id, static_cast<int>(to_idx(src)));
  }
}

void tag_lookup::write(std::filesystem::path const& p) const {
  return cista::write(p, *this);
}

cista::wrapped<tag_lookup> tag_lookup::read(std::filesystem::path const& p) {
  return cista::read<tag_lookup>(p);
}

std::ostream& operator<<(std::ostream& out, tag_lookup const& tags) {
  auto first = true;
  for (auto const [src, tag] : utl::enumerate(tags.src_to_tag_)) {
    if (!first) {
      out << ", ";
    }
    first = false;
    out << src << "=" << tag.view();
  }
  return out;
}

}  // namespace motis