package org.example;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

public class ExpandTsoValueBySlotSize {

  static final String FILE = "C:\\Users\\Leon\\git\\motis\\Auswertung\\Hafas\\QualityAnalysis\\Results\\r-hhlr-a100-hafas-ontrip-false-raptor-gpu-raptor_gpu-tso20.results";
  static final String QUERY = "C:\\Users\\Leon\\git\\motis\\Auswertung\\Hafas\\QualityAnalysis\\Queries\\r-hhlr-a100-hafas-ontrip-false-raptor-gpu-raptor_gpu-tso20.queries";
  //static final String FILE = "C:\\Users\\Leon\\git\\motis\\Auswertung\\QualityAnalysis\\Queries\\hhlr-v100-hafas-pretrip-false-raptor_gpu-tso96.queries";

  static final int SLOT_SIZE = 96;
  static final boolean IS_MCEA = true;


  static final int SLOT_OCC_TIME = 2880 / SLOT_SIZE;

  public static void main(String[] args) throws IOException, ParseException {
    var lines = Files.readAllLines(Path.of(QUERY));
//    var queries = Files.readAllLines(Path.of(QUERY));
//    var parser = new JSONParser();

    var out = new ArrayList<String>();
    var idx = 0;
    for (var line : lines) {
      var new_line = line;
//      var result = (JSONObject)parser.parse(line);
//      var content = (JSONObject)result.get("content");
//      var conns = (JSONArray)content.get("connections");
//      var search_start = 0;
//      var new_line = line;
//
//      var query = (JSONObject)parser.parse(queries.get(idx));
//      var q_content = (JSONObject)query.get("content");
//      var q_start = (JSONObject)q_content.get("start");
//      var q_dep_time = (Long)q_start.get("departure_time");
//
//      for(var c : conns) {
//
//        var conn = (JSONObject)c;
//        var stops = (JSONArray)conn.get("stops");
//
//        var travel_time = 0;
//        var first_stop = (JSONObject) stops.get(0);
//        var first_departure = (JSONObject) first_stop.get("departure");
//        var first_dep_time = (Long) first_departure.get("time");
//
//        var last_stop = (JSONObject) stops.get(stops.size() - 1);
//        var last_arrival = (JSONObject) last_stop.get("arrival");
//        var last_arr_time = (Long) last_arrival.get("time");
//        if(!IS_MCEA) {
//          travel_time = (int) (last_arr_time - first_dep_time);
//        }else {
//          travel_time = (int) (last_arr_time - q_dep_time);
//        }
//
//        var transfers = -1;
//        for(var s : stops) {
//          var stop = (JSONObject)s;
//          var exited = (Boolean)stop.get("exit");
//          if (exited) {
//            ++transfers;
//          }
//        }
//
//        var tso = (Long)conn.get("time_slotted_occupancy");
//
////        tso = tso * SLOT_OCC_TIME;
//
//        var tso_idx = new_line.indexOf("time_slotted_occupancy", search_start);
//        var next_delim = new_line.indexOf(',', tso_idx);
//
//        var up_to_tso = new_line.substring(0, tso_idx);
//        var after_delim = new_line.substring(next_delim);
//
//        new_line = up_to_tso + "time_slotted_occupancy\": " + tso + ", \"travel_time\": " + travel_time + ", \"transfers\": " + transfers + after_delim;
//        search_start = tso_idx + 22;
//      }

      //replace id
      var last_colon = new_line.lastIndexOf(':');
      var up_to_id = new_line.substring(0, last_colon);
      new_line = up_to_id + ": " +  idx + "}";

      ++idx;
      out.add(new_line);
    }

    Files.write(Path.of("C:\\Users\\Leon\\git\\motis\\Auswertung\\Hafas\\QualityAnalysis\\Queries\\r-hhlr-a100-hafas-ontrip-false-raptor-gpu-raptor_gpu-tso20.queries2"), out);
  }
}
