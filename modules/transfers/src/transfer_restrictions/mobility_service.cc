#include "motis/transfers/transfer_restrictions/mobility_service.h"

#include <iostream>

#include "cista/mmap.h"

#include "motis/core/common/logging.h"
#include "motis/transfers/transfer_restrictions/helper.h"

#include "python3.10/Python.h"

#include "utl/parser/buf_reader.h"
#include "utl/parser/cstr.h"
#include "utl/parser/csv_range.h"
#include "utl/parser/line_range.h"
#include "utl/pipes/for_each.h"
#include "utl/progress_tracker.h"

namespace fs = std::filesystem;
namespace ml = motis::logging;
namespace n = ::nigiri;

namespace motis::transfers::restrictions {

void load_mobility_service_availability(fs::path const& path) {
  Py_Initialize();

  // basic imports
  PyRun_SimpleString("import os, sys");
  PyRun_SimpleString("from dotenv import load_dotenv");

  // add sys path to stada api
  PyRun_SimpleString(
      "sys.path.append(os.path.join(os.getcwd(), 'input', 'python', 'dbapi', "
      "'stada'))");

  // import stada function/method
  PyRun_SimpleString(
      "from stations import get_stations_mobility_service_availability_info, "
      "export_multiple_mobility_service_info_to_csv");

  // load dotenv data
  PyRun_SimpleString("load_dotenv()");

  // call stada api; export mobility service availability to csv
  auto const api_call = "export_multiple_mobility_service_info_to_csv( '" +
                        path.string() +
                        "', get_stations_mobility_service_availability_info())";

  PyRun_SimpleString(api_call.c_str());

  Py_Finalize();
}

void update_timetable_with_mobility_service_availability(n::timetable& tt,
                                                         fs::path const& path) {
  // TODO (CARSTEN) UPDATE NIGIRI TIMETABLE STRUCT
  auto const timer = ml::scoped_timer(
      "set mobility service availability as nigiri restriction.");

  struct csv_ms_availability {
    utl::csv_col<utl::cstr, UTL_NAME("name")> name_;
    utl::csv_col<utl::cstr, UTL_NAME("weekday")> weekday_;
    utl::csv_col<utl::cstr, UTL_NAME("from")> from_time_;
    utl::csv_col<utl::cstr, UTL_NAME("to")> to_time_;
  };

  auto name_to_loc_idx_mapping = get_parent_location_name_to_idx_mapping(tt);

  auto file_mmap =
      cista::mmap(path.string().c_str(), cista::mmap::protection::READ);
  auto file_content = file_mmap.view();

  auto const progress_tracker = utl::get_active_progress_tracker();
  progress_tracker->status("Load Mobility Service Restriction")
      .in_high(file_content.size());

  utl::line_range{
      utl::make_buf_reader(file_content, progress_tracker->update_fn())} |
      utl::csv<csv_ms_availability>() |
      utl::for_each([&](csv_ms_availability const& s) {
        std::ignore = s;
        // TODO (CARSTEN) SET MS_AV_ENTRY TO TT (UPDATE TT FIRST)
      });
}

}  // namespace motis::transfers::restrictions
