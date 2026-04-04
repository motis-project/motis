#include "motis/import/tt_import.h"

#include <fstream>

#include "cista/mmap.h"

#include "utl/to_vec.h"

#include "nigiri/loader/assistance.h"
#include "nigiri/loader/load.h"
#include "nigiri/loader/loader_interface.h"
#include "nigiri/clasz.h"
#include "nigiri/common/parse_date.h"
#include "nigiri/shapes_storage.h"
#include "nigiri/timetable.h"
#include "nigiri/timetable_metrics.h"

#include "motis/data.h"
#include "motis/import/route_shapes_import.h"
#include "motis/tag_lookup.h"

namespace n = nigiri;
namespace nl = nigiri::loader;

namespace motis {

namespace fs = std::filesystem;

auto to_clasz_bool_array(
    bool const default_allowed,
    std::optional<std::map<std::string, bool>> const& clasz_allowed) {
  auto a = std::array<bool, n::kNumClasses>{};
  a.fill(default_allowed);
  if (clasz_allowed.has_value()) {
    for (auto const& [clasz, allowed] : *clasz_allowed) {
      a[static_cast<unsigned>(n::to_clasz(clasz))] = allowed;
    }
  }
  return a;
}

tt_import::tt_import(fs::path const& data_path,
                     config const& c,
                     dataset_hashes const& h)
    : task{"tt", data_path, c, {h.tt_, n_version()}},
      keep_routed_shape_data_{
          route_shapes_import::get_keep_routed_shape_data(data_path_, c_, h)} {}

tt_import::~tt_import() = default;

void tt_import::run() {
  auto const& t = *c_.timetable_;

  auto const first_day = n::parse_date(t.first_day_) - std::chrono::days{1};
  auto const interval = n::interval<date::sys_days>{
      first_day, first_day + std::chrono::days{t.num_days_ + 1U}};

  auto assistance = std::unique_ptr<nl::assistance_times>{};
  if (t.assistance_times_.has_value()) {
    auto const f = cista::mmap{t.assistance_times_->generic_string().c_str(),
                               cista::mmap::protection::READ};
    assistance =
        std::make_unique<nl::assistance_times>(nl::read_assistance(f.view()));
  }

  auto shapes = std::unique_ptr<n::shapes_storage>{};
  if (t.with_shapes_) {
    shapes = std::make_unique<n::shapes_storage>(
        data_path_, cista::mmap::protection::WRITE, keep_routed_shape_data_);
  }

  auto tags = cista::wrapped{cista::raw::make_unique<tag_lookup>()};
  auto tt = cista::wrapped{cista::raw::make_unique<n::timetable>(nl::load(
      utl::to_vec(
          t.datasets_,
          [&, src = n::source_idx_t{}](
              std::pair<std::string, config::timetable::dataset> const&
                  x) mutable -> nl::timetable_source {
            auto const& [tag, dc] = x;
            tags->add(src++, tag);
            return {tag,
                    dc.path_,
                    {.link_stop_distance_ = t.link_stop_distance_,
                     .default_tz_ = dc.default_timezone_.value_or(
                         t.default_timezone_.value_or("")),
                     .bikes_allowed_default_ = to_clasz_bool_array(
                         dc.default_bikes_allowed_, dc.clasz_bikes_allowed_),
                     .cars_allowed_default_ = to_clasz_bool_array(
                         dc.default_cars_allowed_, dc.clasz_cars_allowed_),
                     .extend_calendar_ = dc.extend_calendar_,
                     .user_script_ =
                         dc.script_
                             .and_then([](std::string const& path) {
                               if (path.starts_with("\nfunction")) {
                                 return std::optional{path};
                               }
                               return std::optional{std::string{cista::mmap{
                                   path.c_str(), cista::mmap::protection::READ}
                                                                    .view()}};
                             })
                             .value_or("")}};
          }),
      {.adjust_footpaths_ = t.adjust_footpaths_,
       .merge_dupes_intra_src_ = t.merge_dupes_intra_src_,
       .merge_dupes_inter_src_ = t.merge_dupes_inter_src_,
       .max_footpath_length_ = t.max_footpath_length_},
      interval, assistance.get(), shapes.get(), false))};

  tt->write(data_path_ / "tt.bin");
  tags->write(data_path_ / "tags.bin");
  std::ofstream{data_path_ / "timetable_metrics.json"}
      << to_str(n::get_metrics(*tt), *tt);
}

bool tt_import::is_enabled() const { return c_.timetable_.has_value(); }

}  // namespace motis
