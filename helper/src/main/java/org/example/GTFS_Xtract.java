package org.example;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class GTFS_Xtract {

  static class TripServices {
    public TripServices(String trip_id, String service_id) {
      this.trip_id = trip_id;
      this.service_id = service_id;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      if (o == null || getClass() != o.getClass()) return false;
      TripServices that = (TripServices) o;
      return trip_id.equals(that.trip_id) && service_id.equals(that.service_id);
    }

    @Override
    public int hashCode() {
      return Objects.hash(trip_id, service_id);
    }

    public String trip_id;
    public String service_id;
    public int service_no;
  }

  static HashMap<String, Integer> service_map = new HashMap<>();

  static Set<String> extract_trips(JSONObject response) {
    var trip_ids = new HashSet<String>();

    var content = (JSONObject)response.get("content");
    var conns = (JSONArray)content.get("connections");
    for(var c : conns) {
      var conn = (JSONObject)c;
      var trips = (JSONArray) conn.get("trips");
      for (var t : trips) {
        var trip = (JSONObject) t;
        var dbgString = (String) trip.get("debug");


        var trip_id = dbgString.split(":")[0];
        System.out.println("Found trip id: " + trip_id);
        trip_ids.add(trip_id);
      }
    }

    return trip_ids;
  }

  static String find_route_for_trip_id(List<String> tripsFileLines, String trip_id) {

    for(int lineIdx = 1; lineIdx < tripsFileLines.size(); lineIdx++) {
      var line = tripsFileLines.get(lineIdx);
      var lineSplit = line.split(",");
      var lineTripId = lineSplit[2].replace("\"", "");

      if(lineTripId.equals(trip_id)){
        var route_id = lineSplit[0].replace("\"", "");
        System.out.println("Found route id " + route_id + " for trip id " + trip_id);
        return route_id;
      }
    }

    return "";
  }

  static Set<TripServices> find_trips_for_route_id(List<String> tripsFileLines, String route_id) {
    var r = new HashSet<TripServices>();
    for(int lineIdx = 1; lineIdx < tripsFileLines.size(); lineIdx++) {
      var line = tripsFileLines.get(lineIdx);
      var lineSplit = line.split(",");
      var lineRouteId = lineSplit[0].replace("\"", "");

      if(lineRouteId.equals(route_id)) {
        var trip_id = lineSplit[2].replace("\"", "");
        var service_id = lineSplit[1].replace("\"", "");
        System.out.println("Route: " + route_id + "\tTrip Id: " + trip_id + "\tServcie Id: " + service_id);
        r.add(new TripServices(trip_id, service_id));
      }

    }
    return r;
  }

  static int service_line_to_num(String line) {
    var lineSplit = line.split(",");

    var serviceNo = 0;
    for(int i = 1; i < 8; i++) {
      var colVal = Integer.parseInt(lineSplit[i].replace("\"", ""));

      serviceNo = serviceNo | (colVal << (7-i));

    }

    System.out.println("line: " + line + "\t=> " + Integer.toBinaryString(serviceNo));

    return serviceNo;
  }

  static void write_services_file(Path targetServicesFile) {
    var serviceLines = new ArrayList<String>();
    serviceLines.add("service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date");

    for(int i = 0; i < 128; i++) {
      StringBuilder line = new StringBuilder("TA+" + i);

      for (int j = 0; j < 7; j++) {
        var hasOne = ((i & (1 << 6-j)) >> 6-j) == 1;
        if(hasOne)
          line.append(",\"1\"");
        else
          line.append(",\"0\"");
      }

      line.append(",\"20201213\",\"20211211\"");
      serviceLines.add(line.toString());
    }

    try{
      Files.write(targetServicesFile, serviceLines);
    }catch(IOException ex) {
      ex.printStackTrace();
    }

  }

  static void write_reduced_routes_file(Path targetRoutesFile, Path sourceRouteFile, Set<String> routeIds) {
    try{
      var sourceRoutes = Files.readAllLines(sourceRouteFile);

      var targetRouteLines = new ArrayList<String>();
      targetRouteLines.add(sourceRoutes.get(0));

      for (int i = 1; i < sourceRoutes.size(); i++) {
        var sourceRouteId = sourceRoutes.get(i).split(",")[0].replace("\"", "");

        for (String routeId : routeIds) {
          if (sourceRouteId.equals(routeId)) {
            targetRouteLines.add(sourceRoutes.get(i));
          }
        }

      }

      Files.write(targetRoutesFile, targetRouteLines);

    }catch(IOException ex) {
      ex.printStackTrace();
    }
  }

  static void write_necessary_trips(Path targetTripsFile, List<String> sourceTripLines, Set<String> routes) {
    var targetLines = new ArrayList<String>();
    targetLines.add(sourceTripLines.get(0));
    for(int lineIdx = 1; lineIdx < sourceTripLines.size(); lineIdx++) {

      var line = sourceTripLines.get(lineIdx);
      var lineSplit = line.split(",");
      var lineRoute = lineSplit[0].replace("\"", "");

      for(var r_id  : routes) {
        if(lineRoute.equals(r_id)) {
          var targetLine = lineSplit[0]
            + "," + lineSplit[1] + "," + lineSplit[2] + "," + lineSplit[3]
            + "," + lineSplit[4] + "," + lineSplit[5];
          targetLines.add(targetLine);
        }
      }
    }

    try{
      Files.write(targetTripsFile, targetLines);
    }catch(IOException ex) {
      ex.printStackTrace();
    }
  }

  public static void main(String[] args) {
    System.out.println("GTFS-xtract");

    try{
      var out_dir = "data_debug_schedule/gtfs-mod";
      var ssb_sched_path = "data/swiss-gtfs-08-25";

      //1. Read the given responses to extract the routes from
      var responses = Files.readAllLines(Path.of("helper/responses.txt"));
      var parser = new JSONParser();
      var collected_trips = new ArrayList<String>();
      for(var r : responses) {
        var jsonResponse = (JSONObject)parser.parse(r);
        collected_trips.addAll(extract_trips(jsonResponse));
      }

      //2. extract the necessary route ids
      var collected_routes = new HashSet<String>();
      var tripsFileLines = Files.readAllLines(Path.of(ssb_sched_path + "/trips.txt"));
      for(var t : collected_trips) {
        var r_id = find_route_for_trip_id(tripsFileLines, t);
        if(!r_id.isBlank())
          collected_routes.add(r_id);
      }

      write_reduced_routes_file(Path.of(out_dir + "/routes.txt"), Path.of(ssb_sched_path + "/routes.txt"), collected_routes);
      write_necessary_trips(Path.of(out_dir + "/trips.txt"), tripsFileLines, collected_routes);
      //write_services_file(Path.of(args[0] + "/calendar.txt"));

    }catch(IOException | ParseException ex) {
      ex.printStackTrace();
    }

  }
}
