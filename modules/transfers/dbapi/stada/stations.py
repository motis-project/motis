from typing import Dict, List, Any

import csv
import http.client
import json
import os

from dotenv import load_dotenv

__all__ = ["get_stations_mobility_service_availability_info",
           "export_multiple_mobility_service_info_to_csv"]


def get_all_stations() -> http.client.HTTPResponse:
    """ Sends an request to the DB API to get information about all stations in
    the DB network. Outputs the received response as HTTPResponse.

    Returns:
        - A HTTPResponse Object of the HTTPRequest sent to the DB API

    Typical usage example:
        - get_all_stations().read()
        - json.loads(stations.decode("utf-8"))
    """
    conn = http.client.HTTPSConnection("apis.deutschebahn.com")
    conn.request(
        "GET",
        "/db-api-marketplace/apis/station-data/v2/stations",
        headers={
            "DB-Client-Id": os.getenv("CLIENT-ID"),
            "DB-Api-Key": os.getenv("SECRET"),
            "accept": "application/json",
        },
    )
    return conn.getresponse()


def get_stations_mobility_service_availability_info() -> List[Dict[str, Any]]:
    """ Retrieves for each station with mobility service the service times.

    Returns:
        - A list of dictionaries. Each dict with the following structure:  \\
            {  \\
                "name": ...,  \\
                "availability": {  \\
                    "monday": {"fromTime": ..., "toTime": ...},  \\
                    ...  \\
                    "sunday": {"fromTime": ..., "toTime": ...},  \\
                    "holiday": {"fromTime": ..., "toTime": ...}  \\
                }  \\
            }  \\
    """
    service_staff_infos: List[Dict] = []

    stations = get_all_stations().read()
    stations_data: Dict[str, Any] = json.loads(stations.decode("utf-8"))

    for station in stations_data.get("result"):
        service_staff_info = {}

        service_staff_info["name"] = station.get("name")
        has_mobility_service = "localServiceStaff" in station.keys()

        if has_mobility_service:
            service_staff_info["availability"] = station.get(
                "localServiceStaff").get(
                    "availability")

            service_staff_infos.append(service_staff_info)

    return service_staff_infos


# ############ #
# CSV EXPORTER #
# ############ #
def get_csv_station_mobility_service_header_iterable() -> List[str]:
    """ Generates a basic iterable used as csv header for the availability of
    the mobility service.

    Returns:
        - a list of strings (header titles)

    Typical usage example:
        with open(file, "w", encoding="utf-8", newline=""):
            header = get_csv_station_mobility_service_header_iterable()
            csv.writer(file).writerow(header)
    """
    header: List[str] = []

    header.append("name")
    header.append("weekday")
    header.append("from")
    header.append("to")

    return header


def station_mobility_service_availability_info_to_iterables(
        info: Dict[str, Any]) -> List[List[str]]:
    """ Transforms a dictionary (with mobility service availability info) into
    a list of lists.

    Each day as one iterable.

    See:
        - get_stations_service_staff_info

    Returns:
        Returns a list of availability information per day. Therefore, the
        result holds #entries in the availability dictionary elements.

    Typical usage example:
        ```python
        stations_mobility_service_info = 
            get_stations_mobility_service_availability_info()
        with open(x) as f:
            for mobility_service_info in stations_mobility_service_info:
                res = station_mobility_service_availability_info_to_iterables(
                    mobility_service_info)
                writer = csv.writer(f)
                writer.writerows(res)
        ```
    """
    res: List[List[str]] = []
    staff_availability: Dict[str, Dict[str, str]] = info.get("availability")

    for day, availability in staff_availability.items():
        row: List[str] = []

        row.append(info.get("name").replace("\n", ""))
        row.append(day.replace("\n", ""))
        row.append(availability.get("fromTime").replace("\n", ""))
        row.append(availability.get("toTime").replace("\n", ""))

        res.append(row)

    return res


def set_station_mobility_service_availability_csv_header(path: str):
    """ Adds an header to the mobility service information csv file. Removes
    current file content.

    Args:
        - path: path to the csv file to save the mobility service availability
                data.
    """
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(get_csv_station_mobility_service_header_iterable())


def export_single_mobility_service_info_to_csv(
        path: str,
        station_mobility_service_info: Dict[str, Any]):
    """ Saves the mobility availability information of a single station in csv
    format.

    Args:
        - path: path to the csv file to save the mobility service availability
                data.
        - station_mobility_service_info: a dictionary representing the mobility
                service information of an arbitrary station.
    """
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        info_as_iter = station_mobility_service_availability_info_to_iterables(
            station_mobility_service_info)
        writer.writerows(info_as_iter)


def export_multiple_mobility_service_info_to_csv(
        path: str,
        stations_mobility_service_info: List[Dict[str, Any]]):
    """ Saves mobility availability inforamtion of multiple stations in csv
    format.

    Args:
        - path: path to the csv file to save the mobility service availability
                data.
        - stations_mobility_service_info: a list of dictionaries representing
                the mobility service information of an arbitrary number of
                stations.
    """
    set_station_mobility_service_availability_csv_header(path)
    for station_mobility_service_info in stations_mobility_service_info:
        export_single_mobility_service_info_to_csv(
            path,
            station_mobility_service_info)


if __name__ == "__main__":
    load_dotenv()
    stations_mobility_service_info =\
        get_stations_mobility_service_availability_info()
    export_multiple_mobility_service_info_to_csv(
        "db_mobility_service_availability.csv",
        stations_mobility_service_info)
