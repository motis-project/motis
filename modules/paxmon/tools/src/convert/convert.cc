#include <functional>
#include <iostream>
#include <optional>

#include "boost/filesystem.hpp"

#include "utl/enumerate.h"
#include "utl/for_each_line_in_file.h"

#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"

#include "motis/paxmon/csv_writer.h"
#include "motis/paxmon/loader/journeys/journey_access.h"

using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::paxmon;

namespace fs = boost::filesystem;

journey::transport const* get_journey_transport(journey const& j,
                                                std::size_t enter_stop_idx,
                                                std::size_t exit_stop_idx) {
  for (auto const& t : j.transports_) {
    if (t.from_ <= enter_stop_idx && t.to_ >= exit_stop_idx) {
      return &t;
    }
  }
  return nullptr;
}

void for_each_leg(journey const& j,
                  std::function<void(journey::stop const&, journey::stop const&,
                                     extern_trip const&,
                                     journey::transport const*)> const& trip_cb,
                  std::function<void(journey::stop const&,
                                     journey::stop const&)> const& foot_cb) {
  std::optional<std::size_t> exit_stop_idx;
  for (auto const& [stop_idx, stop] : utl::enumerate(j.stops_)) {
    if (stop.exit_) {
      exit_stop_idx = stop_idx;
    }
    if (stop.enter_) {
      auto const jt = get_journey_trip(j, stop_idx);
      if (jt == nullptr) {
        throw std::runtime_error{"invalid journey: trip not found"};
      }
      std::optional<transfer_info> transfer;
      if (exit_stop_idx && *exit_stop_idx != stop_idx) {
        foot_cb(j.stops_.at(*exit_stop_idx), stop);
      }
      trip_cb(stop, j.stops_.at(jt->to_), jt->extern_trip_,
              get_journey_transport(j, stop_idx, jt->to_));
      exit_stop_idx.reset();
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0]
              << " journeys_input.txt journeys_output.csv" << std::endl;
    return 1;
  }

  auto const input_path = argv[1];
  auto const output_path = argv[2];

  if (!fs::is_regular_file(input_path)) {
    std::cerr << "Input file not found: " << input_path << "\n";
    return 1;
  }

  auto writer = csv_writer{output_path};
  writer << "id"
         << "secondary_id"
         << "leg_idx"
         << "leg_type"
         << "from"
         << "to"
         << "enter"
         << "exit"
         << "category"
         << "train_nr"
         << "passengers" << end_row;

  auto write_journey = [&](journey const& j, std::uint64_t primary_id,
                           std::uint64_t secondary_id = 0,
                           std::uint16_t pax = 1) {
    auto leg_idx = 0U;
    for_each_leg(
        j,
        [&](journey::stop const& enter_stop, journey::stop const& exit_stop,
            extern_trip const& et, journey::transport const* transport) {
          writer << primary_id << secondary_id << ++leg_idx << "TRIP"
                 << enter_stop.eva_no_ << exit_stop.eva_no_
                 << enter_stop.departure_.schedule_timestamp_
                 << exit_stop.arrival_.schedule_timestamp_
                 << (transport != nullptr ? transport->category_name_ : "")
                 << et.train_nr_ << pax << end_row;
        },
        [&](journey::stop const& walk_from_stop,
            journey::stop const& walk_to_stop) {
          writer << primary_id << secondary_id << ++leg_idx << "FOOT"
                 << walk_from_stop.eva_no_ << walk_to_stop.eva_no_
                 << walk_from_stop.departure_.schedule_timestamp_
                 << walk_to_stop.arrival_.schedule_timestamp_ << "" << 0 << pax
                 << end_row;
        });
  };

  auto line_nr = 0ULL;
  auto journey_id = 0ULL;
  utl::for_each_line_in_file(input_path, [&](std::string const& line) {
    ++line_nr;
    try {
      auto const res_msg = make_msg(line);
      switch (res_msg->get()->content_type()) {
        case MsgContent_RoutingResponse: {
          auto const res = motis_content(RoutingResponse, res_msg);
          auto const journeys = message_to_journeys(res);
          for (auto const& j : journeys) {
            write_journey(j, ++journey_id);
          }
          break;
        }
        case MsgContent_Connection: {
          auto const res = motis_content(Connection, res_msg);
          auto const j = convert(res);
          write_journey(j, ++journey_id);
          break;
        }
        default: break;
      }
    } catch (std::system_error const& e) {
      std::cerr << "Invalid message: " << e.what() << ": line " << line_nr
                << "\n";
    }
  });

  std::cout << "Converted " << journey_id << " journeys\n";

  return 0;
}
