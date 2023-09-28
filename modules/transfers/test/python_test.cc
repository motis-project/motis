#include "gtest/gtest.h"

#include "python3.10/Python.h"

TEST(python_db_api_example, stations_api_call) {
  // The following is an example of Python usage in C++
  Py_Initialize();
  PyRun_SimpleString("import os, sys");
  PyRun_SimpleString("from dotenv import load_dotenv");
  PyRun_SimpleString(
      "sys.path.append(os.path.join(os.getcwd(), 'modules', 'transfers', "
      "'dbapi', 'stada'))");
  PyRun_SimpleString(
      "from stations import get_stations_mobility_service_availability_info, "
      "export_multiple_mobility_service_info_to_csv");
  PyRun_SimpleString("load_dotenv()");
  PyRun_SimpleString(
      "export_multiple_mobility_service_info_to_csv("
      "'build/data/transfers/db_mobility_service_availability.csv',"
      "get_stations_mobility_service_availability_info())");
  Py_Finalize();
}
