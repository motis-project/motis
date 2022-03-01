#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "boost/algorithm/string.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/program_options.hpp"

#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "utl/erase.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"



using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using namespace motis::isochrone;
using namespace motis::ppr;


std::string query(int id,
                  std::time_t interval_start, std::time_t interval_end,
                  const Position *start_pos, const String *id_str) {
  message_creator fbb;
  auto const start = Position(start_pos->lat(), start_pos->lng());
  auto const interval = Interval(interval_start, interval_end);
  std::vector<Offset<ModeWrapper>> modes_start{CreateModeWrapper(
          fbb, Mode_FootPPR,
          CreateFootPPR(fbb, CreateSearchOptions(fbb, fbb.CreateString("default"),
                                                 15*60))
                  .Union())};
  std::vector<Offset<ModeWrapper>> modes_dest{CreateModeWrapper(
          fbb, Mode_FootPPR,
          CreateFootPPR(fbb, CreateSearchOptions(fbb, fbb.CreateString("default"),
                                                 0))
                  .Union())};
  fbb.create_and_finish(
          MsgContent_IntermodalRoutingRequest,
          CreateIntermodalRoutingRequest(
                  fbb, IntermodalStart_IntermodalOntripStart,
                  CreateIntermodalOntripStart(fbb, &start, interval_start)
                          .Union(),
                  fbb.CreateVector(modes_start), IntermodalDestination_InputStation,
                  CreateInputStation(fbb, fbb.CreateString(id_str), fbb.CreateString("")).Union(),
                  fbb.CreateVector(modes_dest), SearchType_Default,
                  SearchDir_Forward)
                  .Union(),
          "/intermodal");
  auto msg = make_msg(fbb);
  msg->get()->mutate_id(id);

  auto json = msg->to_json();
  utl::erase(json, '\n');
  return json;
}

bool generate_routing_query(const int i, std::pair<msg_ptr, msg_ptr> qr, std::ofstream& routing_queries) {
  if (!std::get<0>(qr) || !std::get<1>(qr)) {
    return false;
  }
  auto const has_error = std::apply(
          [](auto&&... args) {
            auto error = false;
            (([&](msg_ptr const& msg) {
              error = error || msg->get()->content_type() == MsgContent_MotisError;
            })(args),
                    ...);
            return error;
          },
          qr);
  if (has_error) {
    return true;
  }
  auto const& q = std::get<0>(qr);
  auto const& r = std::get<1>(qr);

  auto const& q_msg = motis_content(IsochroneRequest, q);
  auto const& r_msg = motis_content(IsochroneResponse, r);

  std::cout << "query nr:" << i << std::endl;
  for(int j = 0; j < r_msg->stations()->size(); ++j) {
    routing_queries << query(i*1000000 + j, q_msg->departure_time(),
                             q_msg->departure_time()+q_msg->max_travel_time(), q_msg->position(),
                             r_msg->stations()->Get(j)->id()) << "\n";
  }

  return true;
};

int main(int argc, char const** argv) {

  if (argc != 3) {
    std::cout << "Usage: " << argv[0]
              << " {queries.txt} {responses.txt}\n";
    return 0;
  }

  std::ifstream in_q(argv[1]), in_r(argv[2]);

  std::ofstream routing_queries("isochrone_routing_queries.txt");
  std::string line_q, line_r;
  std::map<int, std::pair<msg_ptr, msg_ptr>> pending_msgs;
  while (in_q.peek() != EOF && !in_q.eof() && in_r.peek() != EOF && !in_r.eof()){

    std::getline(in_q, line_q);
    std::getline(in_r, line_r);

    auto const q = make_msg(line_q);
    auto const r = make_msg(line_r);

    std::get<0>(pending_msgs[q->id()]) = q;
    std::get<1>(pending_msgs[r->id()]) = r;

    for (auto const i : {q->id(), r->id()}) {
      auto const it = pending_msgs.find(i);
      if (it == end(pending_msgs)) {
        continue;
      }

      if (generate_routing_query(i, it->second, routing_queries)) {
        pending_msgs.erase(i);
      }
    }



  }
  routing_queries.flush();
  return 0;
}



