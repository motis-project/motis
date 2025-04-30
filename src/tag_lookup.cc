#include <ctime>

#include "motis/tag_lookup.h"

#include "fmt/chrono.h"
#include "fmt/core.h"

#include "cista/io.h"

#include "utl/enumerate.h"
#include "utl/verify.h"

#include "nigiri/rt/frun.h"
#include "nigiri/rt/gtfsrt_resolve_run.h"
#include "nigiri/timetable.h"
#include "utl/parser/split.h"

namespace n = nigiri;

namespace motis {

std::pair<std::string_view, std::string_view> split_tag_id(std::string_view x) {
  auto const first_underscore_pos = x.find('_');
  return first_underscore_pos != std::string_view::npos
             ? std::pair{x.substr(0, first_underscore_pos),
                         x.substr(first_underscore_pos + 1U)}
             : std::pair{std::string_view{}, x};
}

void tag_lookup::add(n::source_idx_t const src, std::string_view str) {
  utl::verify(tag_to_src_.size() == to_idx(src), "invalid tag");
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

std::string tag_lookup::id(nigiri::timetable const& tt,
                           n::rt::run_stop s,
                           n::event_type const ev_type) const {
  if (s.fr_->is_scheduled()) {
    // trip id
    auto const t = s.get_trip_idx(ev_type);
    auto const id_idx = tt.trip_ids_[t].front();
    auto const id = tt.trip_id_strings_[id_idx].view();
    auto const src = tt.trip_id_src_[id_idx];

    // go to first trip stop
    while (s.stop_idx_ > 0 && s.get_trip_idx(n::event_type::kArr) == t) {
      --s.stop_idx_;
    }

    // service date + start time
    auto const [static_transport, utc_start_day] = s.fr_->t_;
    auto const o = tt.transport_first_dep_offset_[static_transport];
    auto const utc_dep =
        tt.event_mam(static_transport, s.stop_idx_, n::event_type::kDep)
            .as_duration();
    auto const gtfs_static_dep = utc_dep + o;
    auto const [day_offset, tz_offset_minutes] =
        n::rt::split_rounded(gtfs_static_dep - utc_dep);
    auto const day = (tt.internal_interval_days().from_ +
                      std::chrono::days{to_idx(utc_start_day)} - day_offset);
    auto const start_hours = gtfs_static_dep / 60;
    auto const start_minutes = gtfs_static_dep % 60;

    return fmt::format("{:%Y%m%d}_{:02}:{:02}_{}_{}", day, start_hours.count(),
                       start_minutes.count(), get_tag(src), id);
  } else {
    auto const id = s.fr_->id();
    auto const time = std::chrono::system_clock::to_time_t(
        (*s.fr_)[0].time(n::event_type::kDep));
    auto const utc = *std::gmtime(&time);
    auto const id_tag = get_tag(id.src_);
    auto const id_id = id.id_;
    return fmt::format("{:04}{:02}{:02}_{:02}:{:02}_{}_{}", utc.tm_year + 1900,
                       utc.tm_mon + 1, utc.tm_mday, utc.tm_hour, utc.tm_min,
                       id_tag, id_id);
  }
}

std::pair<nigiri::rt::run, nigiri::trip_idx_t> tag_lookup::get_trip(
    nigiri::timetable const& tt,
    nigiri::rt_timetable const* rtt,
    std::string_view id) const {
  auto const [date, start_time, tag, trip_id] =
      utl::split<'_', utl::cstr, utl::cstr, utl::cstr, utl::cstr>(id);
  for (auto const rev : {date, start_time, tag, trip_id}) {
    utl::verify(rev.valid(), "invalid tripId {}", id);
  }
  auto td = transit_realtime::TripDescriptor{};
  td.set_start_date(date.view());
  td.set_start_time(start_time.view());
  td.set_trip_id(std::string_view{
      trip_id.str,
      static_cast<std::size_t>(id.data() + id.size() - trip_id.str)});

  return n::rt::gtfsrt_resolve_run({}, tt, rtt, get_src(tag.view()), td);
}

nigiri::location_idx_t tag_lookup::get_location(nigiri::timetable const& tt,
                                                std::string_view s) const {
  auto const [tag, id] = split_tag_id(s);
  auto const src = get_src(tag);
  try {
    return tt.locations_.location_id_to_idx_.at({{id}, src});
  } catch (...) {
    throw utl::fail(
        R"(could not find timetable location "{}", tag="{}", id="{}", src={})",
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