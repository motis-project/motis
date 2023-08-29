#include "motis/intermodal/eval/commands.h"

#include <cstring>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "boost/program_options.hpp"

#include "utl/to_vec.h"

#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/core/conv/trip_conv.h"
#include "motis/core/journey/check_journey.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using namespace flatbuffers;

namespace motis::intermodal::eval {

struct mins {
  int min_improvement;
  journey min;
};

/*function getImprovement(a: Connection, b: Connection, weights: number[]) {
  const criteriaA = a.getCriteria3();
  const criteriaB = b.getCriteria3();

  let dist = 0.0;
  let improvement = 0.0;

  for (let i = 0; i != weights.length; ++i) {
    const weightedA = criteriaA[i] * weights[i];
    const weightedB = criteriaB[i] * weights[i];
    const criterionDist = weightedA - weightedB;

    dist += Math.pow(criterionDist, 2);
    if (criterionDist < 0) {
      improvement += Math.pow(criterionDist, 2);
    }
  }

  dist = Math.sqrt(dist);
  improvement = Math.sqrt(improvement);

  if (improvement == 0) {
    return 0;
  }

  const p = 30.0;
  const q = 0.1;

  return Math.log2(Math.pow(improvement, 2) / dist) * (Math.atan(p * (dist - q)) + Math.PI / 2.0);
}*/

mins getMinImprovement(journey conn, std::vector<journey> x_cons, int[] weights) {
  auto min_improvement = 100;//Number.MAX_VALUE;
  journey min;

  for(auto const x : x_cons) {
    auto improvement = get_improvement(conn, x, weights);
    if(improvement < min_improvement) {
      min_improvement = improvement;
      min = x;
    }
  }
  mins all_min_vals = {min_improvement, min};
  return all_min_vals;
}

double get_improvement(std::vector<journey> cons_a, std::vector<journey> cons_b, int[] weights, bool lr) {//weights: number[]) {
  if(cons_a.size() == 0 && cons_b.size() == 0) {
    return 0.0;
  } else if (cons_a.size() == 0) {
    return 1;//Number.MIN_VALUE;
  } else if (cons_b.size() == 0) {
    return 1;//Number.MAX_VALUE;
  }

  if(lr) {
    std::vector<journey> a_copy(cons_a);
    std::vector<journey> b_copy(cons_b);
  }
  else {
    std::vector<journey> a_copy(cons_b);
    std::vector<journey> b_copy(cons_a);
  }
  double improvement = 0.0;

  while(!a_copy.empty()) {
    auto max_improvement_a = -1.0;//Number.MAX_VALUE;
    auto a_max = 0
    auto b_min = 0;

    for(auto const a : a_copy) {
      // let {minImprovement, min} = get_min_improvement(a, b_copy, weights);
      // if(minImprovement > max_improvement_a) {
      // max_improvement_a = minImprovement;
      // a_max = a;
      // b_min = min;
      //}
      //});
    }

    improvement += max_improvement_a;
    //aCopy.splice(aCopy.indexOf(maxA), 1);
    //bCopy.push(maxA);
  }
 return improvement;
}


double improvement_check(int id, std::vector<msg_ptr> const& responses,
                       std::vector<std::string> const& files,
                       bool local, bool pretty_print) {
  assert(responses.size() == files.size());
  assert(responses.size() > 1);
  auto const file_count = files.size();
  double improvement = 0.0;

  auto const res = utl::to_vec(responses, [](auto const& m) {
    return motis_content(RoutingResponse, m);
  });

  auto const refcons_without_filter = message_to_journeys(res[0]);
  auto const cons_with_filter = message_to_journeys(res[1]);

  // Hilfsfunktionen
  std::ostringstream journey_errors;
  auto const report_journey_error = [&](bool) -> std::ostream& {
    return journey_errors;
  };
  auto const check_journeys = [&](auto const file_idx,
                                  std::vector<journey> const& journeys) {
    for (auto const& j : journeys) {
      if (!check_journey(j, report_journey_error)) {
        std::cout << "Broken journey (id: " << id  << "): " << journey_errors.str() << std::endl;
        journey_errors.str("");
      }
    }
  };

  check_journeys(0, refcons_without_filter);
  check_journeys(1, cons_with_filter);

  auto con_size_without = refcons_without_filter.size();
  auto con_size_with = cons_with_filter.size();

  int[] weights = {1, 1, 30};
  const l_r_impro = get_improvement(refcons_without_filter, cons_with_filter, weights, true);
  const r_l_impro = get_improvement(refcons_without_filter, cons_with_filter, weights, false);
  improvement = l_r_impro - r_l_impro;

  return improvement;
}

int filter_compare(int argc, char const** argv) {
  using namespace motis::intermodal::eval;

  bool help = false;
  bool utc = false;
  bool local = true;
  bool pretty_print = false;
  std::vector<std::string> filenames;
  po::options_description desc("Filter Comparator");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help), "show help")
      ("utc,u", po::bool_switch(&utc), "print timestamps in UTC")
      ("local,l", po::bool_switch(&local), "print timestamps in local time")
      ("pretty,p", po::bool_switch(&pretty_print), "pretty-print json files")
      ("responses", po::value<std::vector<std::string>>(&filenames)->multitoken(), "response files");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (help || filenames.size() != 2) {
    fmt::print("{}\n", desc);
    if (filenames.size() != 2) {
      fmt::print("only {} filenames given, ==2 required: {}\n",
                 filenames.size(), filenames);
    }
    return 0;
  }

  if (utc) {
    local = false;
  }

  auto in_files = utl::to_vec(filenames, [](std::string const& filename) {
    return std::ifstream{filename};
  });
  auto const file_count = in_files.size(); // 2
  std::vector<std::unordered_map<int, msg_ptr>> queued_queries(file_count);
  std::vector<std::unordered_map<int, msg_ptr>> queued_msgs(file_count);

  // Zwei Files:
  // Jeweils mehrere Lines = Messages
  // Messages sind dann journeys mit mehreren Connections.
  // Jeweils vergleichen: Message bzw journey mit id 1 aus File A
  //                 mit: Message bzw journey mit id 1 aus File B

  // TODO:
  // 1. queued msgs von File A und von File B mit ids speichern
  // 2. Dann Funktion aufrufen die jeweils die Message mit id 1 aus den queues nimmt
  // 3. Dort zu journeys und connection sets umbauen, Felix Tool -> Wert
  // 4. Wert printen, jeweils für alle ids

  auto msg_count = 0;
  auto non_empty_msg_count = 0;
  auto errors = 0;
  auto done = false;
  std::unordered_set<int> read_id;
  read_id.reserve(file_count);
  while(!done) {
    read_id.clear();
    done = true;
    for(auto i = 0UL; i < file_count; ++i) {
      auto& in = in_files[i];
      if (in.peek() != EOF && !in.eof()) {
        std::string line;
        std::getline(in, line);
        auto const m = make_msg(line);
        if(m->get()->content_type() == MsgContent_RoutingResponse) {
          queued_msgs[i][m->id()] = m;
          read_id.insert(m->id());
          done = false;
        }
      }
    }
    for(auto const& id : read_id) {
      std::vector<msg_ptr> msgs;
      msgs.reserve(file_count);
      for(auto i = 0UL; i < file_count; ++i) {
        auto const& q_responses = queued_msgs[i];
        auto it = q_responses.find(id);
        if (it == end(q_responses)) {
          break;
        } else {
          msgs.emplace_back(it->second);
        }
      }
      if(msgs.size() == file_count) {
        ++msg_count;
        auto improvement = improvement_check(id, msgs, filenames, local, pretty_print);
        if(improvement == -1.0) {
          ++errors;
        }
        if(motis_content(RoutingResponse, msgs[0])->connections()->size() == 0U) {
          ++non_empty_msg_count;
        }
        for(auto& q : queued_msgs) {
          q.erase(id);
        }
      }
    }
  }


  return errors;
}
} // namespace motis::intermodal::eval