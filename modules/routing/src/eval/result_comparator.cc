#include "motis/routing/eval/commands.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "motis/core/common/unixtime.h"
#include "motis/module/message.h"
#include "motis/protocol/RoutingRequest_generated.h"
#include "motis/routing/eval/comparator/response.h"
#include "motis/routing/eval/get_stat.h"

using namespace motis;
using namespace motis::routing;
using namespace motis::module;
using namespace motis::eval;
using namespace motis::eval::comparator;

namespace motis::routing::eval {

template <typename T>
char get_relation_symbol(T const& u1, T const& u2) {
  if (u1 == u2) {
    return '=';
  } else if (u1 < u2) {
    return '<';
  } else {
    return '>';
  }
}

std::string format_time(unixtime t, bool local_time) {
  constexpr auto const TIME_FORMAT = "%d.%m. %H:%M";
  std::time_t conv = t;
  std::ostringstream out;
  out << std::put_time(
      local_time ? std::localtime(&conv) : std::gmtime(&conv),  // NOLINT
      TIME_FORMAT);
  return out.str();
}

void print(journey_meta_data const& con) {
  auto const format_duration = [](int seconds) {
    auto const total_minutes = seconds / 60;
    auto const days = total_minutes / (60 * 24);
    auto const hours = (total_minutes - days * 60 * 24) / 60;
    auto const minutes = total_minutes - days * 60 * 24 - hours * 60;

    std::stringstream ss;
    if (days != 0) {
      ss << days << "d";
    }
    if (hours != 0) {
      ss << " " << std::right << std::setw(2) << hours << "h";
    }
    if (minutes != 0) {
      ss << " " << std::right << std::setw(2) << minutes << "min";
    }
    return ss.str();
  };

  std::cout << std::right << std::setw(13) << format_duration(con.duration_)  //
            << " [" << format_time(con.get_departure_time(), true) << " - "
            << format_time(con.get_arrival_time(), true) << "]\t"  //
            << std::setw(5) << con.transfers_;
}

void print_empty() {
  std::cout << std::setw(27) << std::left << "-"
            << "\t" << std::setw(5) << "-";
}

bool print_differences(response const& r1, response const& r2,
                       RoutingRequest const* req, int id,
                       bool print_only_second_empty) {
  if (r1.valid() && r2.valid() && r1.connections_ == r2.connections_) {
    return true;
  }

  if (!r1.valid()) {
    std::cout << "FIRST CONTAINS INVALID CONNECTION\n";
  }
  if (!r2.valid()) {
    std::cout << "SECOND CONTAINS INVALID CONNECTION\n";
  }
  if (print_only_second_empty &&
      (r1.connections_.empty() || !r2.connections_.empty())) {
    return false;
  }

  std::cout << "ERROR [id = " << id << "]: ";

  if (req->start_type() == Start_PretripStart) {
    auto const interval_begin =
        static_cast<time_t>(reinterpret_cast<PretripStart const*>(req->start())
                                ->interval()
                                ->begin());
    auto const interval_end = static_cast<time_t>(
        reinterpret_cast<PretripStart const*>(req->start())->interval()->end());
    auto const begin_tm = *std::localtime(&interval_begin);  // NOLINT
    auto const end_tm = *std::localtime(&interval_end);  // NOLINT
    std::cout << std::put_time(&begin_tm, "%FT%TZ") << " - "
              << std::put_time(&end_tm, "%FT%TZ") << "\n";
  } else if (req->start_type() == Start_OntripStationStart) {
    auto const departure_time = static_cast<time_t>(
        reinterpret_cast<OntripStationStart const*>(req->start())
            ->departure_time());
    auto const departure_tm = *std::localtime(&departure_time);  // NOLINT
    std::cout << std::put_time(&departure_tm, "%FT%TZ") << "\n";
  }
  if (r1.connections_.size() != r2.connections_.size()) {
    std::cout << "#con1 = " << r1.connections_.size() << ", "
              << "#con2 = " << r2.connections_.size() << " ";
  } else {
    std::cout << "#con = " << r1.connections_.size() << " ";
  }
  if (static_cast<bool>(get_stat(r1.r_, "routing", "max_label_quit"))) {
    std::cout << "--- con1: MAX_LABEL_QUIT ---\n";
  }
  if (static_cast<bool>(get_stat(r2.r_, "routing", "max_label_quit"))) {
    std::cout << "--- con2: MAX_LABEL_QUIT ---\n";
  }

  auto end1 = end(r1.connections_);
  auto end2 = end(r2.connections_);
  {
    std::cout << "\n";
    auto it1 = begin(r1.connections_);
    auto it2 = begin(r2.connections_);
    int i = 0;
    while (true) {
      bool stop1 = false, stop2 = false;

      std::cout << "   " << std::setw(2) << i++ << ":  ";

      if (it1 != end1) {
        print(*it1);
        ++it1;
      } else {
        print_empty();
        stop1 = true;
      }

      std::cout << "\t\t";

      if (it2 != end2) {
        print(*it2);
        ++it2;
      } else {
        print_empty();
        stop2 = true;
      }

      std::cout << "\n";

      if (stop1 && stop2) {
        break;
      }
    }
    std::cout << "\n";
  }

  std::vector<bool> matches1(r1.connections_.size()),
      matches2(r2.connections_.size());
  bool con1_dominates = false, con2_dominates = false;
  unsigned con_count1 = 0;
  for (auto it1 = begin(r1.connections_); it1 != end1; ++it1, ++con_count1) {
    unsigned con_count2 = 0;
    for (auto it2 = begin(r2.connections_); it2 != end2; ++it2, ++con_count2) {
      if (*it1 == *it2) {
        matches2[con_count2] = matches1[con_count1] = true;
        continue;
      }

      std::string domination_info;
      if (it1->dominates(*it2)) {
        domination_info = "\tFIRST DOMINATES \t";
        con1_dominates = true;
      } else if (it2->dominates(*it1)) {
        domination_info = "\tSECOND DOMINATES\t";
        con2_dominates = true;
      } else {
        continue;
      }

      std::cout << "  " << std::setw(2) << con_count1 << " vs " << std::setw(2)
                << con_count2 << domination_info << "\n";
    }
  }

  std::cout << "\n  -> connections in FIRST with no match in SECOND: ";
  bool match1 = false;
  for (auto i = 0UL; i < matches1.size(); i++) {
    if (!matches1[i]) {
      std::cout << i << " ";
      match1 = true;
    }
  }
  if (!match1) {
    std::cout << "-";
  }

  std::cout << "\n  -> connections in SECOND with no match in FIRST: ";
  bool match2 = false;
  for (auto i = 0UL; i < matches2.size(); i++) {
    if (!matches2[i]) {
      std::cout << i << " ";
      match2 = true;
    }
  }
  if (!match2) {
    std::cout << "-";
  }

  if (con1_dominates && !con2_dominates) {
    std::cout << "\n  -> total domination by FIRST\n\n\n";
  } else if (con2_dominates && !con1_dominates) {
    std::cout << "\n  -> total domination by SECOND\n\n\n";
  } else {
    std::cout << "\n  -> no total domination\n\n\n";
  }

  return false;
}

void write_file(std::string const& content, std::string const& filename) {
  std::ofstream out(filename);
  out << content << "\n";
}

struct statistics {
  int total() const { return matches_ + mismatches_ + errors_; }
  int matches_{0}, mismatches_{0};
  int errors_{0};
};

bool analyze_result(int i, std::tuple<msg_ptr, msg_ptr, msg_ptr> const& res,
                    std::ofstream& failed_queries, statistics& stats,
                    bool print_only_second_empty) {
  if (!std::get<0>(res) || !std::get<1>(res) || !std::get<2>(res)) {
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
      res);
  if (has_error) {
    ++stats.errors_;
    return true;
  }

  auto const& q = std::get<0>(res);
  auto const& r1 = std::get<1>(res);
  auto const& r2 = std::get<2>(res);

  auto const ontrip_start = motis_content(RoutingRequest, q)->start_type() ==
                            Start_OntripStationStart;

  if (print_differences(
          response(motis_content(RoutingResponse, r1), ontrip_start),
          response(motis_content(RoutingResponse, r2), ontrip_start),
          motis_content(RoutingRequest, q), i, print_only_second_empty)) {
    ++stats.matches_;
  } else {
    ++stats.mismatches_;
    failed_queries << q->to_json(true) << "\n";
    failed_queries.flush();
    write_file(r1->to_json(true),
               "fail_responses/" + std::to_string(i) + "_1.json");
    write_file(r2->to_json(true),
               "fail_responses/" + std::to_string(i) + "_2.json");
    write_file(q->to_json(true), "fail_queries/" + std::to_string(i) + ".json");
  }

  return true;
}

int compare(int argc, char const** argv) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0]
              << " {results.txt I} {results.txt II} {queries.txt}\n";
    return 0;
  }

  bool print_only_second_empty = false;

  statistics stats;
  std::ifstream in1(argv[1]), in2(argv[2]), inq(argv[3]);
  std::ofstream failed_queries("failed_queries.txt");
  std::string line1, line2, lineq;
  std::map<int, std::tuple<msg_ptr, msg_ptr, msg_ptr>> pending_msgs;
  while (in1.peek() != EOF && !in1.eof() && in2.peek() != EOF && !in2.eof() &&
         inq.peek() != EOF && !inq.eof()) {
    std::getline(in1, line1);
    std::getline(in2, line2);
    std::getline(inq, lineq);

    auto const r1 = make_msg(line1);
    auto const r2 = make_msg(line2);
    auto const q = make_msg(lineq);

    std::get<0>(pending_msgs[q->id()]) = q;
    std::get<1>(pending_msgs[r1->id()]) = r1;
    std::get<2>(pending_msgs[r2->id()]) = r2;

    for (auto const i : {q->id(), r1->id(), r2->id()}) {
      auto const it = pending_msgs.find(i);
      if (it == end(pending_msgs)) {
        continue;
      }

      if (analyze_result(i, it->second, failed_queries, stats,
                         print_only_second_empty)) {
        pending_msgs.erase(i);
      }
    }
  }

  if (!pending_msgs.empty()) {
    std::cout << "warning: " << pending_msgs.size()
              << " unmatched queries/responses:\n";
    for (auto const& m : pending_msgs) {
      auto const& id = m.first;
      auto const& v = m.second;

      std::cout << "  id=" << id << ": "  //
                << "query_set=" << std::boolalpha
                << static_cast<bool>(std::get<0>(v)) << " "
                << "res1_set=" << std::boolalpha
                << static_cast<bool>(std::get<1>(v)) << " "
                << "res2_set=" << std::boolalpha
                << static_cast<bool>(std::get<2>(v)) << "\n";
    }
  }

  std::cout << "\nStatistics:\n"
            << "  #matches = " << stats.matches_ << "/" << stats.total() << "\n"
            << "  #mismatches  = " << stats.mismatches_ << "/" << stats.total()
            << "\n"
            << "  #invalid = " << stats.errors_ << "/" << stats.total() << "\n";

  return (stats.mismatches_ == 0 && stats.errors_ == 0) ? 0 : 1;
}

}  // namespace motis::routing::eval
