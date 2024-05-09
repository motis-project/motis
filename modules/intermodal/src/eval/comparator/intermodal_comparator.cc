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

#include "fmt/ranges.h"

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

constexpr auto const TIME_FORMAT = "%d.%m. %H:%M";

std::string format_time(unixtime t, bool local_time) {
  std::time_t const conv = t;
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

query_type_t get_query_type(msg_ptr const& msg) {
  switch (msg->get()->content_type()) {
    case MsgContent_IntermodalRoutingRequest: {
      auto const req = motis_content(IntermodalRoutingRequest, msg);
      return (req->start_type() == IntermodalStart_IntermodalPretripStart ||
              req->start_type() == IntermodalStart_PretripStart)
                 ? query_type_t::PRETRIP
                 : (req->search_dir() == SearchDir_Forward
                        ? query_type_t::ONTRIP_FWD
                        : query_type_t::ONTRIP_BWD);
    }
    case MsgContent_RoutingRequest: {
      auto const req = motis_content(RoutingRequest, msg);
      return req->start_type() == routing::Start_PretripStart
                 ? query_type_t::PRETRIP
                 : (req->search_dir() == SearchDir_Forward
                        ? query_type_t::ONTRIP_FWD
                        : query_type_t::ONTRIP_BWD);
    }
    default:
      throw utl::fail("unsupported query type {}",
                      EnumNameMsgContent(msg->get()->content_type()));
  }
}

void print_query(std::ostream& out, msg_ptr const& msg, bool const local) {
  auto const print_pretrip_start = [&](routing::PretripStart const* start) {
    out << start->station()->id()->view() << " @ ["
        << format_time(start->interval()->begin(), local) << ", "
        << format_time(start->interval()->end(), local) << "]";
  };

  auto const print_ontrip_station_start =
      [&](routing::OntripStationStart const* start) {
        out << start->station()->id()->view() << " @ "
            << format_time(start->departure_time(), local);
      };

  auto const print_ontrip_train_start =
      [&](routing::OntripTrainStart const* start) {
        out << start->station()->id()->view() << " @ "
            << format_time(start->arrival_time(), local) << " @ "
            << to_extern_trip(start->trip()).to_str();
      };

  switch (msg->get()->content_type()) {
    case MsgContent_IntermodalRoutingRequest: {
      auto const req = motis_content(IntermodalRoutingRequest, msg);
      out << (req->search_dir() == SearchDir_Forward ? "FWD" : "BWD") << " ";
      switch (req->start_type()) {
        case intermodal::IntermodalStart_IntermodalPretripStart: {
          auto const start =
              reinterpret_cast<intermodal::IntermodalPretripStart const*>(
                  req->start());
          out << "(" << start->position()->lat() << ", "
              << start->position()->lng() << ") @ ["
              << format_time(start->interval()->begin(), local) << ", "
              << format_time(start->interval()->end(), local) << "]";
          return;
        }
        case intermodal::IntermodalStart_IntermodalOntripStart: {
          auto const start =
              reinterpret_cast<intermodal::IntermodalOntripStart const*>(
                  req->start());
          out << "(" << start->position()->lat() << ", "
              << start->position()->lng() << ") @ "
              << format_time(start->departure_time(), local);
          return;
        }
        case intermodal::IntermodalStart_PretripStart:
          return print_pretrip_start(
              reinterpret_cast<routing::PretripStart const*>(req->start()));
        case intermodal::IntermodalStart_OntripStationStart:
          return print_ontrip_station_start(
              reinterpret_cast<routing::OntripStationStart const*>(
                  req->start()));
        case intermodal::IntermodalStart_OntripTrainStart:
          return print_ontrip_train_start(
              reinterpret_cast<routing::OntripTrainStart const*>(req->start()));
        default:
          throw utl::fail("unsupported intermodal start type {}",
                          EnumNameIntermodalStart(req->start_type()));
      }
    }
    case MsgContent_RoutingRequest: {
      auto const req = motis_content(RoutingRequest, msg);
      out << (req->search_dir() == SearchDir_Forward ? "FWD" : "BWD") << " ";
      switch (req->start_type()) {
        case routing::Start_PretripStart:
          return print_pretrip_start(
              reinterpret_cast<routing::PretripStart const*>(req->start()));
        case routing::Start_OntripStationStart:
          return print_ontrip_station_start(
              reinterpret_cast<routing::OntripStationStart const*>(
                  req->start()));
        case routing::Start_OntripTrainStart:
          return print_ontrip_train_start(
              reinterpret_cast<routing::OntripTrainStart const*>(req->start()));
        default:
          throw utl::fail("unsupported routing start type {}",
                          EnumNameStart(req->start_type()));
      }
    }

    default:
      throw utl::fail("unsupported query type {}",
                      EnumNameMsgContent(msg->get()->content_type()));
  }
}

bool check(int id, std::vector<msg_ptr> const& responses,
           std::vector<msg_ptr> const& queries,
           std::vector<std::string> const& response_files,
           std::vector<std::string> const& query_files,
           std::vector<int>& file_errors, fs::path const& fail_path, bool local,
           bool pretty_print) {
  assert(responses.size() == response_files.size());
  assert(responses.size() > 1);
  auto const file_count = response_files.size();
  auto match = true;
  std::unordered_set<int> failed_files;
  auto const query_type = get_query_type(queries.at(0));

  auto const res = utl::to_vec(responses, [](auto const& m) {
    return motis_content(RoutingResponse, m);
  });

  auto const fail = [&](auto const file_idx) -> std::ostream& {
    if (match) {
      std::cout << "\nMismatch at id=" << id << ", query={";
      print_query(std::cout, queries[0], local);
      std::cout << "}:\n";
      match = false;
    }
    failed_files.insert(file_idx);
    std::cout << "  " << response_files[file_idx] << ": ";
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
        fail(file_idx) << "Broken journey" << journey_errors.str() << std::endl;
        journey_errors.str("");
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
              << '\n';
    }

    for (auto const& ref_con : ref) {
      if (!has_connection(r, ref_con, query_type)) {
        fail(i) << "Connection with duration=" << std::setw(4)
                << ref_con.duration_ << ", transfers=" << ref_con.transfers_
                << ", accessibility=" << std::setw(2) << ref_con.accessibility_
                << ", start=" << format_time(departure_time(ref_con), local)
                << ", end=" << format_time(arrival_time(ref_con), local)
                << " missing" << '\n';
      }
    }

    for (auto const& con : r) {
      if (!has_connection(ref, con, query_type)) {
        fail(i) << "Connection with duration=" << std::setw(4) << con.duration_
                << ", transfers=" << con.transfers_
                << ", accessibility=" << std::setw(2) << con.accessibility_
                << ", start=" << format_time(departure_time(con), local)
                << ", end=" << format_time(arrival_time(con), local)
                << " unexpected" << '\n';
      }
    }

    check_journeys(i, r);
  }

  auto const write_file = [&](auto const file_idx) {
    if (fail_path.empty()) {
      return;
    }

    auto const jf = pretty_print ? json_format::DEFAULT_FLATBUFFERS
                                 : json_format::SINGLE_LINE;

    std::ofstream out{
        (fail_path / fmt::format("{}_{}.json", std::to_string(id),
                                 file_identifier(response_files[file_idx])))
            .string()};
    out << responses[file_idx]->to_json(jf) << std::endl;

    if (!queries.empty()) {
      std::ofstream query_out{
          (fail_path / fmt::format("{}_{}.json", std::to_string(id),
                                   file_identifier(query_files[file_idx])))
              .string()};
      query_out << queries[file_idx]->to_json(jf) << std::endl;
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
  po::options_description desc("Intermodal Comparator");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help), "show help")
      ("utc,u", po::bool_switch(&utc), "print timestamps in UTC")
      ("local,l", po::bool_switch(&local), "print timestamps in local time")
      ("pretty,p", po::bool_switch(&pretty_print), "pretty-print json files")
      ("queries", po::value<std::vector<std::string>>(&query_paths)->multitoken(),
       "query files if failed queries should be written")
      ("responses", po::value<std::vector<std::string>>(&filenames)->multitoken(), "response files")
      ("fail", po::value<std::string>(&fail_dir)->default_value("fail"),
          "output directory for different responses (empty to disable)");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (help || filenames.size() < 2) {
    fmt::print("{}\n", fmt::streamed(desc));
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
  auto non_empty_msg_count = 0;
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
          if (m->get()->content_type() == MsgContent_IntermodalRoutingRequest ||
              m->get()->content_type() == MsgContent_RoutingRequest) {
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
        auto success = check(id, msgs, queries, filenames, query_paths,
                             file_errors, fail_path, local, pretty_print);
        if (!success) {
          ++errors;
        }
        if (motis_content(RoutingResponse, msgs[0])->connections()->size() ==
            0U) {
          ++non_empty_msg_count;
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
              << " messages are missing in some results:" << '\n';
    for (auto const& id : unmatched_msgs) {
      std::cout << " query id " << id << " missing in:";
      for (auto file_idx = 0UL; file_idx < file_count; ++file_idx) {
        auto const& response = queued_msgs[file_idx];
        if (auto const it = response.find(id); it == end(response)) {
          std::cout << " " << filenames[file_idx];
        } else {
          std::ofstream out{
              (fail_path / fmt::format("{}_{}.json", std::to_string(id),
                                       file_identifier(filenames[file_idx])))
                  .string()};
          out << it->second->to_json(pretty_print
                                         ? json_format::DEFAULT_FLATBUFFERS
                                         : json_format::SINGLE_LINE)
              << '\n';
        }

        if (with_queries) {
          auto const& query = queued_queries[file_idx];
          if (auto const it = query.find(id); it != end(query)) {
            std::ofstream query_out{
                (fail_path /
                 fmt::format("{}_{}.json", std::to_string(id),
                             file_identifier(query_paths[file_idx])))
                    .string()};
            query_out << it->second->to_json(
                             pretty_print ? json_format::DEFAULT_FLATBUFFERS
                                          : json_format::SINGLE_LINE)
                      << '\n';
          }
        }
      }
      std::cout << '\n';
    }
    std::cout << "\n\n\n";
  }

  std::cout << "Queries where responses don't match: " << errors << "/"
            << msg_count << " (non-empty: " << non_empty_msg_count
            << ", non-empty-error-rate: "
            << static_cast<int>(
                   (static_cast<double>(errors) / non_empty_msg_count) * 100)
            << "%)" << '\n';
  std::cout << "Mismatches by file:" << '\n';
  std::cout << "  " << filenames[0] << ": Used as reference" << '\n';
  for (auto i = 1UL; i < filenames.size(); ++i) {
    std::cout << "  " << filenames[i] << ": " << file_errors[i] << '\n';
  }

  if (errors > 0 && !fail_path.empty()) {
    std::cout << "\nResponses that don't match written to: "
              << fail_path.string() << '\n';
  }

  return (errors == 0 && unmatched_msgs.empty()) ? 0 : 1;
}

}  // namespace motis::intermodal::eval
