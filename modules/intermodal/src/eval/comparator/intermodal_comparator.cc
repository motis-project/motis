#include "motis/intermodal/eval/commands.h"

#include <cstring>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "boost/filesystem.hpp"
#include "boost/program_options.hpp"

#include "utl/to_vec.h"

#include "geo/latlng.h"
#include "geo/webmercator.h"

#include "motis/core/common/unixtime.h"
#include "motis/core/schedule/time.h"
#include "motis/core/access/time_access.h"
#include "motis/core/journey/check_journey.h"
#include "motis/core/journey/journey.h"
#include "motis/core/journey/message_to_journeys.h"
#include "motis/module/message.h"
#include "motis/bootstrap/dataset_settings.h"
#include "motis/bootstrap/motis_instance.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace motis;
using namespace motis::bootstrap;
using namespace motis::module;
using namespace motis::routing;
using namespace motis::intermodal;
using namespace flatbuffers;

namespace motis::intermodal::eval {

constexpr auto const TIME_FORMAT = "%d.%m. %H:%M";

std::string format_time(unixtime t, bool local_time) {
  std::time_t conv = t;
  std::ostringstream out;
  out << std::put_time(
      local_time ? std::localtime(&conv) : std::gmtime(&conv),  // NOLINT
      TIME_FORMAT);
  return out.str();
}

enum class query_type_t { PRETRIP, ONTRIP_FWD, ONTRIP_BWD };

std::istream& operator>>(std::istream& in, query_type_t& type) {
  std::string token;
  in >> token;
  if (token == "pretrip") {
    type = query_type_t::PRETRIP;
  } else if (token == "ontrip_fwd") {
    type = query_type_t::ONTRIP_FWD;
  } else if (token == "ontrip_bwd") {
    type = query_type_t::ONTRIP_BWD;
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, query_type_t const& type) {
  switch (type) {
    case query_type_t::PRETRIP: out << "pretrip"; break;
    case query_type_t::ONTRIP_FWD: out << "ontrip_fwd"; break;
    case query_type_t::ONTRIP_BWD: out << "ontrip_bwd"; break;
  }
  return out;
}

inline unixtime departure_time(journey const& j) {
  return j.stops_.front().departure_.timestamp_;
}

inline unixtime arrival_time(journey const& j) {
  return j.stops_.back().arrival_.timestamp_;
}

bool has_connection(std::vector<journey> const& connections,
                    journey const& ref_con, query_type_t const query_type) {
  switch (query_type) {
    case query_type_t::PRETRIP:
      return std::any_of(
          begin(connections), end(connections), [&](auto const& con) {
            return con.duration_ == ref_con.duration_ &&
                   con.transfers_ == ref_con.transfers_ &&
                   con.accessibility_ == ref_con.accessibility_ &&
                   departure_time(con) == departure_time(ref_con) &&
                   arrival_time(con) == arrival_time(ref_con);
          });
    case query_type_t::ONTRIP_FWD:
      return std::any_of(
          begin(connections), end(connections), [&](auto const& con) {
            return con.transfers_ == ref_con.transfers_ &&
                   con.accessibility_ == ref_con.accessibility_ &&
                   arrival_time(con) == arrival_time(ref_con);
          });
    case query_type_t::ONTRIP_BWD:
      return std::any_of(
          begin(connections), end(connections), [&](auto const& con) {
            return con.transfers_ == ref_con.transfers_ &&
                   con.accessibility_ == ref_con.accessibility_ &&
                   departure_time(con) == departure_time(ref_con);
          });
    default: return false;
  }
}

std::string file_identifier(std::string const& filename) {
  return fs::path{filename}.stem().string();
}

bool check(int id, std::vector<msg_ptr> const& msgs,
           std::vector<msg_ptr> const& queries,
           std::vector<std::string> const& files,
           std::vector<std::string> const& query_files,
           std::vector<int>& file_errors, fs::path const& fail_path, bool local,
           query_type_t const query_type, bool pretty_print) {
  assert(msgs.size() == files.size());
  assert(msgs.size() > 1);
  auto const file_count = files.size();
  auto match = true;
  std::unordered_set<int> failed_files;

  auto const res = utl::to_vec(
      msgs, [](auto const& m) { return motis_content(RoutingResponse, m); });

  auto const fail = [&](auto const file_idx) -> std::ostream& {
    if (match) {
      std::cout << "\nMismatch at query id " << id << ":\n";
      match = false;
    }
    failed_files.insert(file_idx);
    std::cout << "  " << files[file_idx] << ": ";
    return std::cout;
  };

  std::ostringstream journey_errors;
  auto const report_journey_error = [&](bool) -> std::ostream& {
    return journey_errors;
  };

  auto const check_journeys = [&](auto const file_idx,
                                  std::vector<journey> const& journeys) {
    for (auto const& j : journeys) {
      if (!check_journey(j, report_journey_error)) {
        fail(file_idx) << "Broken journey" << std::endl;
      }
    }
  };

  auto const ref = message_to_journeys(res[0]);
  auto const ref_cons = ref.size();

  check_journeys(0, ref);

  for (auto i = 1UL; i < file_count; ++i) {
    auto const r = message_to_journeys(res[i]);
    if (r.size() != ref_cons) {
      fail(i) << "Expected " << ref_cons << " connections, has " << r.size()
              << std::endl;
    }

    for (auto const& ref_con : ref) {
      if (!has_connection(r, ref_con, query_type)) {
        fail(i) << "Connection with duration=" << std::setw(4)
                << ref_con.duration_ << ", transfers=" << ref_con.transfers_
                << ", accessibility=" << std::setw(2) << ref_con.accessibility_
                << ", start=" << format_time(departure_time(ref_con), local)
                << ", end=" << format_time(arrival_time(ref_con), local)
                << " missing" << std::endl;
      }
    }

    for (auto const& con : r) {
      if (!has_connection(ref, con, query_type)) {
        fail(i) << "Connection with duration=" << std::setw(4) << con.duration_
                << ", transfers=" << con.transfers_
                << ", accessibility=" << std::setw(2) << con.accessibility_
                << ", start=" << format_time(departure_time(con), local)
                << ", end=" << format_time(arrival_time(con), local)
                << " unexpected" << std::endl;
      }
    }

    check_journeys(i, r);
  }

  auto const write_file = [&](auto const file_idx) {
    if (fail_path.empty()) {
      return;
    }

    std::ofstream out{
        (fail_path / fmt::format("{}_{}.json", std::to_string(id),
                                 file_identifier(files[file_idx])))
            .string()};
    out << msgs[file_idx]->to_json(!pretty_print) << std::endl;

    if (!queries.empty()) {
      std::ofstream query_out{
          (fail_path / fmt::format("{}_{}.json", std::to_string(id),
                                   file_identifier(query_files[file_idx])))
              .string()};
      query_out << queries[file_idx]->to_json(!pretty_print) << std::endl;
    }
  };

  if (!match) {
    write_file(0);
    for (auto const& file_idx : failed_files) {
      file_errors[file_idx]++;
      if (file_idx != 0) {
        write_file(file_idx);
      }
    }
  }

  return match;
}

int compare(int argc, char const** argv) {
  using namespace motis::intermodal::eval;

  bool help = false;
  bool utc = false;
  bool local = false;
  bool pretty_print = false;
  std::vector<std::string> filenames;
  std::vector<std::string> query_paths;
  std::string fail_dir;
  query_type_t query_type{query_type_t::PRETRIP};
  po::options_description desc("Intermodal Comparator");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help), "show help")
      ("utc,u", po::bool_switch(&utc), "print timestamps in UTC")
      ("local,l", po::bool_switch(&local), "print timestamps in local time")
      ("pretty,p", po::bool_switch(&pretty_print), "pretty-print json files")
      ("input", po::value<std::vector<std::string>>(&filenames)->multitoken(), "response files")
      ("fail", po::value<std::string>(&fail_dir)->default_value("fail"),
          "output directory for different responses (empty to disable)")
      ("queries", po::value<std::vector<std::string>>(&query_paths)->multitoken(),
       "query files if failed queries should be written")
      ("type,t",
          po::value<query_type_t>(&query_type)->default_value(query_type),
          "query type: pretrip|ontrip_fwd|ontrip_bwd");
  // clang-format on

  for (int i = 0; i != argc; ++i) {
    fmt::print("{}: {}\n", i, argv[i]);
  }

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (help || filenames.size() < 2) {
    fmt::print("{}\n", desc);
    if (filenames.size() < 2) {
      fmt::print("only {} filenames given, >=2 required: {}\nquery_paths: {}\n",
                 filenames.size(), filenames, query_paths);
    }
    return 0;
  }

  if (utc) {
    local = false;
  }

  auto const fail_path = fs::path{fail_dir};
  if (!fail_path.empty()) {
    fs::create_directories(fail_path);
  }

  auto in_files = utl::to_vec(filenames, [](std::string const& filename) {
    return std::ifstream{filename};
  });
  auto const file_count = in_files.size();

  auto in_query_files = utl::to_vec(
      query_paths,
      [](std::string const& filename) { return std::ifstream{filename}; });
  utl::verify(
      in_query_files.empty() || in_query_files.size() == file_count,
      "query paths ({}) should be either empty or match the response file "
      "count ({})",
      in_query_files.size(), file_count);

  auto const with_queries = !in_query_files.empty();
  std::vector<std::unordered_map<int, msg_ptr>> queued_queries(file_count);
  std::vector<std::unordered_map<int, msg_ptr>> queued_msgs(file_count);
  std::vector<int> file_errors(file_count);

  auto msg_count = 0;
  auto errors = 0;
  auto done = false;
  std::unordered_set<int> read;
  read.reserve(file_count);
  while (!done) {
    read.clear();
    done = true;
    for (auto i = 0UL; i < file_count; ++i) {
      if (with_queries) {
        auto& query_in = in_query_files[i];
        if (query_in.peek() != EOF && !query_in.eof()) {
          std::string line;
          std::getline(query_in, line);
          auto const m = make_msg(line);
          if (m->get()->content_type() == MsgContent_IntermodalRoutingRequest) {
            queued_queries[i][m->id()] = m;
            read.insert(m->id());
            done = false;
          }
        }
      }

      auto& in = in_files[i];
      if (in.peek() != EOF && !in.eof()) {
        std::string line;
        std::getline(in, line);
        auto const m = make_msg(line);
        if (m->get()->content_type() == MsgContent_RoutingResponse) {
          queued_msgs[i][m->id()] = m;
          read.insert(m->id());
          done = false;
        }
      }
    }
    for (auto const& id : read) {
      std::vector<msg_ptr> msgs;
      std::vector<msg_ptr> queries;
      msgs.reserve(file_count);
      for (auto i = 0UL; i < file_count; ++i) {
        {
          auto const& q_responses = queued_msgs[i];
          auto it = q_responses.find(id);
          if (it == end(q_responses)) {
            break;
          } else {
            msgs.emplace_back(it->second);
          }
        }

        if (with_queries) {
          auto const& q_queries = queued_queries[i];
          auto it = q_queries.find(id);
          if (it == end(q_queries)) {
            break;
          } else {
            queries.emplace_back(it->second);
          }
        }
      }
      if (msgs.size() == file_count &&
          (!with_queries || queries.size() == file_count)) {
        ++msg_count;
        auto success =
            check(id, msgs, queries, filenames, query_paths, file_errors,
                  fail_path, local, query_type, pretty_print);
        if (!success) {
          ++errors;
        }
        for (auto& q : queued_msgs) {
          q.erase(id);
        }
      }
    }
  }

  std::cout << "\n\n\n";

  std::set<int> unmatched_msgs;
  for (auto const& q : queued_msgs) {
    for (auto const& m : q) {
      unmatched_msgs.insert(m.first);
    }
  }
  if (!unmatched_msgs.empty()) {
    std::cout << unmatched_msgs.size()
              << " messages are missing in some results:" << std::endl;
    for (auto const& id : unmatched_msgs) {
      std::cout << " query id " << id << " missing in:";
      for (auto file_idx = 0UL; file_idx < file_count; ++file_idx) {
        const auto& q = queued_msgs[file_idx];
        if (q.find(id) == end(q)) {
          std::cout << " " << filenames[file_idx];
        }
      }
      std::cout << std::endl;
    }
    std::cout << "\n\n\n";
  }

  std::cout << "Queries where responses don't match: " << errors << "/"
            << msg_count << std::endl;
  std::cout << "Mismatches by file:" << std::endl;
  std::cout << "  " << filenames[0] << ": Used as reference" << std::endl;
  for (auto i = 1UL; i < filenames.size(); ++i) {
    std::cout << "  " << filenames[i] << ": " << file_errors[i] << std::endl;
  }

  if (errors > 0 && !fail_path.empty()) {
    std::cout << "\nResponses that don't match written to: "
              << fail_path.string() << std::endl;
  }

  return (errors == 0 && unmatched_msgs.empty()) ? 0 : 1;
}

}  // namespace motis::intermodal::eval
