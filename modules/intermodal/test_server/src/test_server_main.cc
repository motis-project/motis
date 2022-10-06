#include "motis/bootstrap/motis_instance.h"
#include <iostream>
#include "net/stop_handler.h"
#include "test_server_impl.h"


int main(int argc, char** argv) {
  std::cout << "Welcome to the ondemand test server! \n";
  std::vector<std::string> server_arguments;
  std::string load = "low";
  std::string area = "0";
  for (int i = 1; i < argc; i = i+2) {
    std::string arg = argv[i];
    if ((arg == "-h") || (arg == "--help")) {
      std::cerr << "Usage of " << argv[0] << "\n"
                << "Options: (default value in [] brackets)\n"
                << "\t-h,--help\tShow this help message\n"
                << "\t-l,--load\tPossible Options: low, medium, high "
                   "To simulate the compute time for the server [low]\n"
                << "\t-a,--area\tPossible Options: 0,1,2,3 "
                   "On which area configuration should the server run? [swiss complete: 0]\n"
                << std::endl;
      return 0;
    }
    else if ((arg == "-l") || (arg == "--load")) {
      if (i + 1 < argc) {
        server_arguments.emplace_back(argv[i+1]);
        load = argv[i+1];
      } else {
        std::cerr << "--load option requires one argument: "
                     "'low' or 'medium' or 'high' \t"
                  << "Or use --help to show usage information\n"
                  << std::endl;
        return 1;
      }
    }
    else if((arg == "-a") || (arg == "--area")) {
      if (i + 1 < argc) {
        server_arguments.emplace_back(argv[i+1]);
        area = argv[i+1];
      } else {
        std::cerr << "--area option requires one argument: "
                     "'0' or '1' or '2' or '3' for an area in which the ondemand"
                     "transport is working \t"
                  << "Or use --help to show usage information\n"
                  << std::endl;
        return 1;
      }
    }
    else {
      std::cout << "This is NOT an option. \n"
                   "Use --help to to show usage information and possible server options. \n"
                   "All default values will be used.";
    }
  }

  std::cout << "\nOptions in use: \n"
               "load option: " << load << "\n"
               "area option: " << area << "\n\n";

  motis::bootstrap::motis_instance new_instance;
  motis::intermodal::test_server servertest(new_instance.runner_.ios(), server_arguments);

  boost::system::error_code ectest;
  servertest.listen_tome("127.0.0.1", "9000", ectest);
  if (ectest) {
    std::cout << "unable to start testserver: " << ectest.message() << "\n";
    return 1;
  }
  new_instance.runner_.run(1, false);
  return 0;
}

