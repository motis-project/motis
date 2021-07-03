#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

#include "boost/algorithm/string.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"
#include "boost/program_options.hpp"

#include "utl/erase.h"

#include "motis/core/access/time_access.h"
#include "motis/module/message.h"



using namespace flatbuffers;
using namespace motis;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::isochrone;


std::string query(std::string const& target, Start const start_type, int id,
                  std::time_t interval_start, std::time_t interval_end,
                  const String *from_eva, const String *to_eva,
                  SearchDir const dir) {
  message_creator fbb;
  auto const interval = Interval(interval_start, interval_end);
  fbb.create_and_finish(
          MsgContent_RoutingRequest,
          CreateRoutingRequest(
                  fbb, start_type,
                  start_type == Start_PretripStart
                  ? CreatePretripStart(
                          fbb,
                          motis::routing::CreateInputStation(fbb, fbb.CreateString(from_eva),
                                             fbb.CreateString("")),
                          &interval)
                          .Union()
                  : CreateOntripStationStart(
                          fbb,
                          motis::routing::CreateInputStation(fbb, fbb.CreateString(from_eva),
                                             fbb.CreateString("")),
                          interval_start)
                          .Union(),
                  motis::routing::CreateInputStation(fbb, fbb.CreateString(to_eva),
                                     fbb.CreateString("")),
                  SearchType_Default, dir, fbb.CreateVector(std::vector<Offset<Via>>()),
                  fbb.CreateVector(std::vector<Offset<AdditionalEdgeWrapper>>()), false, false)
                  .Union(),
          target);
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
    routing_queries << query("/routing", Start_OntripStationStart, i*1000000 + j, q_msg->departure_time(),
                             q_msg->departure_time()+q_msg->max_travel_time(), q_msg->station()->id(),
                             r_msg->stations()->Get(j)->id(),
                             SearchDir_Forward) << "\n";
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



