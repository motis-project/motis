#include "motis/intermodal/eval/commands.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <cfloat>

#include "boost/program_options.hpp"
#include "utl/to_vec.h"
#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/access/time_access.h"
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

#define PI 3.141592653589793238462643383279

struct mins {
  double min_improvement_;
  journey min_;
};

struct improv_pair {
  double improvement_;
  int id_;
};

std::vector<improv_pair> normalize(const std::vector<improv_pair>& to_normalize) {
  std::vector<improv_pair> normalized;
  improv_pair norm{};
  double max = -DBL_MAX;
  double min = DBL_MAX;
  for(auto const v : to_normalize) {
    if(v.improvement_ > max) {
      max = v.improvement_;
    }
    if(v.improvement_ < min) {
      min = v.improvement_;
    }
  }
  double denominator = max - min;
  if(denominator == 0) {
    denominator = 1;
  }
  for(auto const impr : to_normalize) {
    double impr_norm = (impr.improvement_ - min) / denominator;
    norm = {impr_norm, impr.id_};
    normalized.emplace_back(norm);
  }
  return normalized;
}

double get_improvement(journey a, journey b, std::vector<int> weights) {
  // criteria 1 dep_time
  unixtime criteria1_a = a.stops_.at(0).departure_.timestamp_;
  unixtime criteria1_b = b.stops_.at(0).departure_.timestamp_;
  //printf("DEBUG: A Criteria 1: Depature: %lld \n", criteria1_a);
  //printf("DEBUG: B Criteria 1: Depature: %lld \n", criteria1_b);
  // criteria 2 arr_time
  unixtime criteria2_a = a.stops_.at(a.stops_.size()-1).arrival_.timestamp_;
  unixtime criteria2_b = b.stops_.at(b.stops_.size()-1).arrival_.timestamp_;
  //printf("DEBUG: A Criteria 2: Arrival: %lld \n", criteria2_a);
  //printf("DEBUG: B Criteria 2: Arrival: %lld \n", criteria2_b);
  // criteria 3 transfers
  int64_t citeria3_a = a.transfers_;
  int64_t citeria3_b = b.transfers_;
  //printf("DEBUG: A Criteria 3: Transfers: %lld \n", citeria3_a);
  //printf("DEBUG: B Criteria 3: Transfers: %lld \n", citeria3_b);
  // all critera
  std::vector<unixtime> all_criteria_a = {criteria1_a, criteria2_a, citeria3_a};
  std::vector<unixtime> all_criteria_b = {criteria1_b, criteria2_b, citeria3_b};
  double dist = 0.0;
  double improvement = 0.0;

  for(int i = 0; i != weights.size(); ++i) {
    auto const weighted_a = all_criteria_a.at(i) * weights.at(i);
    auto const weighted_b = all_criteria_b.at(i) * weights.at(i);
    auto const criterion_dist = weighted_a - weighted_b;

    dist += std::pow(criterion_dist, 2);
    if(criterion_dist < 0) {
      improvement += std::pow(criterion_dist, 2);
    }
  }

  dist = std::sqrt(dist);
  improvement = std::sqrt(improvement);

  if (improvement == 0) {
    return 0.0;
  }

  double const p = 30.0;
  double const q = 0.1;
  //(log2(std::pow(improvement, 2) / dist) * ((atan(p * (dist - q)) + PI) / 2.0));
  double value1 = log2(std::pow(improvement, 2) / dist);
  double value2 = (atan(p * (dist - q)) + PI) / 2.0;
  return value1 * value2;
}

mins get_min_improvement(const journey& conn, const std::vector<journey>& x_cons, const std::vector<int>& weights) {
  auto min_improvement = DBL_MAX;
  journey min;

  for(auto const& x : x_cons) {
    auto improvement = get_improvement(conn, x, weights);
    if(improvement < min_improvement) {
      min_improvement = improvement;
      min = x;
    }
  }
  mins all_min_vals = {min_improvement, min};
  return all_min_vals;
}

double eval_improvement(const std::vector<journey>& cons_a, const std::vector<journey>& cons_b, const std::vector<int>& weights) {
  if(cons_a.empty() && cons_b.empty()) {
    return 0.0;
  } else if (cons_a.empty()) {
    return -DBL_MAX;
  } else if (cons_b.empty()) {
    return DBL_MAX;
  }

  std::vector<journey> a_copy(cons_a);
  std::vector<journey> b_copy(cons_b);
  double improvement = 0.0;

  while(!a_copy.empty()) {
    double max_improvement_a = -DBL_MAX;
    journey a_max;
    journey b_min;
    int ix_a_max = 0;

    for(int i = 0; i != a_copy.size(); ++i) {
      journey a = a_copy.at(i);
      mins min_values = get_min_improvement(a, b_copy, weights);
      if(min_values.min_improvement_ > max_improvement_a) {
        max_improvement_a = min_values.min_improvement_;
        a_max = a;
        ix_a_max = i;
        b_min = min_values.min_;
      }
    }

    improvement += max_improvement_a;
    a_copy.erase(a_copy.begin() + ix_a_max);
    b_copy.emplace_back(a_max);
  }
 return improvement;
}


double improvement_check(int id, std::vector<msg_ptr> const& responses,
                       std::vector<std::string> const& files) {
  assert(responses.size() == files.size());
  assert(responses.size() > 1);
  auto const file_count = files.size();
  double improvement = 0.0;

  auto const res = utl::to_vec(responses, [](auto const& m) {
    return motis_content(RoutingResponse, m);
  });

  auto const refcons_without_filter = message_to_journeys(res[0]);
  auto const cons_with_filter = message_to_journeys(res[1]);

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

  //auto con_size_without = refcons_without_filter.size();
  //auto con_size_with = cons_with_filter.size();
  //std::cout << "There are " << con_size_without << " journeys from File " << files.at(0) << std::endl;
  //std::cout << "There are " << con_size_with << " journeys from File " << files.at(1) << std::endl;

  std::vector<int> weights = {1, 1, 30};
  auto const l_r_impro = eval_improvement(refcons_without_filter, cons_with_filter, weights);
  auto const r_l_impro = eval_improvement(cons_with_filter, refcons_without_filter, weights);
  improvement = l_r_impro - r_l_impro;

  return improvement;
}

int filter_compare(int argc, char const** argv) {
  using namespace motis::intermodal::eval;
  bool help = false;
  std::vector<std::string> filenames;
  po::options_description desc("Filter Comparator");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help), "show help")
      ("files,f", po::value<std::vector<std::string>>(&filenames)->multitoken(), "files to compare");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (help || filenames.size() != 2) {
    fmt::print("{}\n", desc);
    fmt::print(" This comparator takes two files. The first file should consist of reference journeys.\n "
        "The second file should consist of results, which used the station filter or other aspects one wants to check against.\n "
        "The comparator will compare each journey and and compute an \"improvement\". \n\n");
    if (filenames.size() != 2) {
      fmt::print("only {} file(s) given, ==2 required: {}\n",
                 filenames.size(), filenames);
    }
    return 0;
  }

  auto in_files = utl::to_vec(filenames, [](std::string const& filename) {
    return std::ifstream{filename};
  });
  auto const file_count = in_files.size();
  std::vector<std::unordered_map<int, msg_ptr>> queued_queries(file_count);
  std::vector<std::unordered_map<int, msg_ptr>> queued_msgs(file_count);

  auto msg_count = 0;
  auto non_empty_msg_count = 0;
  auto errors = 0;
  auto without_partner = 0;
  auto done = false;
  std::vector<improv_pair> all_improvements;
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
        auto improvement = improvement_check(id, msgs, filenames);
        improv_pair improv = {improvement, id};
        all_improvements.emplace_back(improv);
        if(improvement < 0.0) {
          ++errors;
        }
        if(motis_content(RoutingResponse, msgs[0])->connections()->size() == 0U) {
          ++non_empty_msg_count;
        }
        for(auto& q : queued_msgs) {
          q.erase(id);
        }
      } else {
        ++without_partner;
        std::cout << "The journey with id " << id << " does not have a counterpart!" << std::endl;
      }
    }
  }

  // normalize and print results
  std::vector<improv_pair> normalized_improv = normalize(all_improvements);
  std::cout << "[---Results:---]" << std::endl;
  std::cout << "non-emtpy-msg-count: " << non_empty_msg_count << std::endl;
  std::cout << "msg-count: " << msg_count << std::endl;
  std::cout << "journeys without counterpart: " << without_partner << std::endl;
  std::cout << "errors: " << errors << std::endl;
  std::cout << "Not normalized results: " << std::endl;
  for(auto const p : all_improvements) {
    std::cout << "ID: " << p.id_ << "\t Improvement: " << p.improvement_ << std::endl;
  }
  std::cout << "Normalized results: " << std::endl;
  for(auto const pn : normalized_improv) {
    std::cout << "ID: " << pn.id_ << "\t Improvement: " << pn.improvement_ << std::endl;
  }

  return errors;
}
} // namespace motis::intermodal::eval