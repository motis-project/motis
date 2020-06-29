#include "motis/paxmon/capacity.h"

#include <cstdint>

#include "utl/parser/buf_reader.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/file.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"

namespace motis::paxmon {

namespace {

struct row {
  utl::csv_col<utl::cstr, UTL_NAME("category")> category_;
  utl::csv_col<std::uint32_t, UTL_NAME("train_nr")> train_nr_;
  utl::csv_col<std::uint16_t, UTL_NAME("capacity")> capacity_;
};

};  // namespace

capacity_map_t load_capacities(std::string const& capacity_file) {
  auto buf = utl::file(capacity_file.data(), "r").content();
  auto const file_content = utl::cstr{buf.data(), buf.size()};
  auto map = mcd::hash_map<train_name, std::uint16_t>{};

  utl::line_range<utl::buf_reader>{file_content}  //
      | utl::csv<row>()  //
      | utl::for_each([&](auto&& row) {
          auto const tn =
              train_name{row.category_.val().view(), row.train_nr_.val()};
          map[tn] = row.capacity_.val();
        });

  return map;
}

std::uint16_t guess_capacity(schedule const& sched,
                             light_connection const& lc) {
  auto const& category_name =
      sched.categories_[lc.full_con_->con_info_->family_]->name_;

  if (category_name == "ICE") {
    return 340;
  } else if (category_name == "IC") {
    return 140;
  } else if (category_name == "EC") {
    return 260;
  } else if (category_name == "RB") {
    return 100;
  } else if (category_name == "TGV") {
    return 509;
  } else if (category_name == "THA") {
    return 371;
  } else if (category_name == "RJ") {
    return 404;
  } else if (category_name == "NJ" || category_name == "EN") {
    return 194;
  }

  switch (lc.full_con_->clasz_) {
    case 0: return 340;  // high speed
    case 1: return 140;  // long range
    case 2: return 194;  // night trains
    case 3:  // fast local trains
    case 4: return 100;  // local trains
    case 5: return 175;  // metro
    case 6: return 50;  // subway
    case 7: return 60;  // street-car
    case 8: return 30;  // bus
    default: return 60;  // other
  }
}

std::uint16_t get_capacity(schedule const& sched, capacity_map_t const& map,
                           light_connection const& lc) {
  auto const& con_info = lc.full_con_->con_info_;
  auto const tn = train_name{sched.categories_[con_info->family_]->name_,
                             con_info->train_nr_};
  auto const it = map.find(tn);
  return it != end(map) ? it->second : guess_capacity(sched, lc);
}

}  // namespace motis::paxmon
