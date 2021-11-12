package org.example.data;

import org.example.Utils;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;

public abstract class AbstractConnection<ST extends Stop> {
  public abstract ST newStop(int idx, JSONObject st);

  public AbstractConnection(JSONObject conn) {
    var stops = (JSONArray) conn.get("stops");
    var trips = (JSONArray) conn.get("trips");
    if(conn.containsKey("max_occupancy"))
      this.moc = (Long) conn.get("max_occupancy");
    else{
      var ucriteris = (JSONObject)conn.get("ucriteria");
      this.moc = (Long) ucriteris.get("occupancy_max");
    }

    this.stops = new ArrayList<>();
    for (int i = 0, stopsSize = stops.size(); i < stopsSize; i++) {
      var st = (JSONObject) stops.get(i);
      this.stops.add(newStop(i, st));
    }

    this.unix_arr_time = this.stops.get(this.stops.size() - 1).unix_arr_time;
    this.unix_dep_time = this.stops.get(0).unix_dep_time;
    this.motis_arr_time = Utils.unix_to_motis_time(this.unix_arr_time);
    this.motis_dep_time = Utils.unix_to_motis_time(this.unix_dep_time);

    this.duration = Duration.between(LocalDateTime.ofEpochSecond(this.unix_dep_time, 0, ZoneOffset.UTC),
      LocalDateTime.ofEpochSecond(this.unix_arr_time, 0, ZoneOffset.UTC));
    this.durationStr = String.format("%3d:%02d:%02d", duration.toHours(), duration.toMinutesPart(), duration.toSecondsPart());
    this.tripCount = trips.size();
  }

  public String departureFmt() {
    return DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.ofEpochSecond(unix_dep_time, 0, ZoneOffset.UTC));
  }

  public String arrivalFmt() {
    return DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(LocalDateTime.ofEpochSecond(unix_arr_time, 0, ZoneOffset.UTC));
  }

  public final long unix_arr_time;
  public final long unix_dep_time;
  public final long motis_arr_time;
  public final long motis_dep_time;
  public final Duration duration;
  public final String durationStr;
  public final long tripCount;
  public final long moc;
  public final List<ST> stops;
}
