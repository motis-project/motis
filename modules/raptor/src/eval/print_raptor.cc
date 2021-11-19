#include "motis/raptor/eval/commands.h"

#include <fstream>
#include <iomanip>
#include <iostream>

#include "motis/core/access/trip_access.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"
#include "motis/raptor/raptor_timetable.h"
#include "motis/raptor/get_raptor_timetable.h"
#include "motis/raptor/print_raptor.h"

#include "conf/configuration.h"
#include "conf/options_parser.h"

#include "motis/protocol/RoutingResponse_generated.h"

#include "version.h"

using namespace flatbuffers;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::raptor;
using motis::module::make_msg;

namespace motis::raptor::eval {

struct print_raptor_options : public conf::configuration {
  print_raptor_options() : configuration{"Print options"} {
    param(in_path_, "in", "Input file path");
  }

  std::string in_path_;
};

void print(schedule const& sched, raptor_meta_info const& meta_info,
           journey const& j) {
  std::cout << "Journey with TR: " << j.trips_.size()
            << ";\tMOC: " << j.max_occupancy_ << ";\tDuration: " << j.duration_
            << ";\n=========================================================\n";

  // iterate over the trips in the journey and validate the occupancy values to
  //   the expectation
  std::string prev_arr_eva{j.stops_[0].eva_no_};
  auto prev_arr_time = j.stops_[0].departure_.timestamp_;
  bool had_fp = false;
  for (int idx = 0, size = j.trips_.size(); idx < size; ++idx) {
    auto const& j_trip = j.trips_[idx];
    auto const s_trip = get_trip(sched, j_trip.extern_trip_);
    auto const s_trip_dbg = s_trip->dbg_.str();

    auto const j_from_stop = j.stops_[j_trip.from_];
    auto const j_to_stop = j.stops_[j_trip.to_];

    if (prev_arr_eva != j_from_stop.eva_no_) {
      // there is a Footpath in between
      std::cout << "     From:\ts_id: " << std::setw(6)
                << +meta_info.eva_to_raptor_id_.at(prev_arr_eva)
                << ";\teva: " << prev_arr_eva << ";\tDep: " << std::setw(7)
                << +unix_to_motistime(sched.schedule_begin_, prev_arr_time)
                << " (" << prev_arr_time << ");\n"
                << "     Using: Footpath\n"
                << "     To:\ts_id: " << std::setw(6)
                << +meta_info.eva_to_raptor_id_.at(j_from_stop.eva_no_)
                << ";\teva: " << j_from_stop.eva_no_
                << ";\tArr: " << std::setw(7)
                << +unix_to_motistime(sched.schedule_begin_,
                                      j_from_stop.arrival_.timestamp_)
                << " (" << j_from_stop.arrival_.timestamp_ << ");\n";
    }

    std::cout << std::setw(3) << idx << ": "
              << "From:\ts_id: " << std::setw(6)
              << +meta_info.eva_to_raptor_id_.at(j_from_stop.eva_no_)
              << ";\teva: " << j_from_stop.eva_no_ << ";\tDep: " << std::setw(7)
              << +unix_to_motistime(sched.schedule_begin_,
                                    j_from_stop.departure_.timestamp_)
              << " (" << j_from_stop.departure_.timestamp_ << ");\n"
              << "     Using: " << s_trip_dbg
              << meta_info.route_mapping_.str(s_trip_dbg) << ";\n"
              << "     To:\ts_id: " << std::setw(6)
              << +meta_info.eva_to_raptor_id_.at(j_to_stop.eva_no_)
              << ";\teva: " << j_to_stop.eva_no_ << ";\tArr: " << std::setw(7)
              << +unix_to_motistime(sched.schedule_begin_,
                                    j_to_stop.arrival_.timestamp_)
              << " (" << j_to_stop.arrival_.timestamp_ << ");\n";

    prev_arr_eva = j_to_stop.eva_no_;
    prev_arr_time = j_to_stop.arrival_.timestamp_;
    had_fp = false;
  }
}

int print_raptor(int argc, const char** argv) {
  print_raptor_options print_opt;
  dataset_settings dataset_opt;

  try {
    conf::options_parser parser({&dataset_opt, &print_opt});
    parser.read_command_line_args(argc, argv, false);

    if (parser.help()) {
      std::cout << "\n\tMOTIS " << short_version() << "\n\n";
      parser.print_help(std::cout);
      return 0;
    } else if (parser.version()) {
      std::cout << "MOTIS " << long_version() << "\n";
      return 0;
    }

    parser.read_configuration_file(true);
    parser.print_used(std::cout);
  } catch (std::exception const& e) {
    std::cout << "options error: " << e.what() << "\n";
    return 1;
  }

  motis_instance instance;
  instance.import(module_settings{}, dataset_opt,
                  import_settings{{dataset_opt.dataset_}});

  auto const& sched = instance.sched();
  auto const [meta_info, tt] = get_raptor_timetable(sched);

  print_theoretical_moc_figures(*tt);

  std::ifstream in{print_opt.in_path_.c_str()};
  in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  std::string json;
  while (!in.eof() && in.peek() != EOF) {
    std::getline(in, json);
    auto const message = make_msg(json);

    if (message->get()->content_type() != MsgContent_RoutingResponse) {
      throw std::exception{
          "Found message with content type other than RoutingResponse!"};
    }

    auto const res = motis_content(RoutingResponse, message);

    std::cout << "\nQuery ID: " << message->id() << "\n";
    std::cout << "=====================================================\n";

    auto const journeys = message_to_journeys(res);
    for (auto const& j : journeys) {
      print(sched, *meta_info, j);
    }
  }

  return 0;
}

}