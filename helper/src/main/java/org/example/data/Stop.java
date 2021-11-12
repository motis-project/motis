package org.example.data;

import org.example.Utils;
import org.json.simple.JSONObject;

import java.time.LocalDateTime;
import java.time.ZoneOffset;

public class Stop {
  public Stop(int idx, JSONObject stop) {
    this.idx = idx;
    var station = (JSONObject) stop.get("station");
    var arrival = (JSONObject) stop.get("arrival");
    var departure = (JSONObject) stop.get("departure");

    this.eva_no = (String) station.get("id");
    this.name = (String) station.get("name");

    var arr_time = arrival.get("time");
    if(arr_time != null) {
      this.unix_arr_time = (Long) arr_time;
      if (this.unix_arr_time > 0)
        this.motis_arr_time = Utils.unix_to_motis_time(this.unix_arr_time);
      else
        this.motis_arr_time = 0;
    }else {
      this.unix_arr_time = 0;
      this.motis_arr_time = 0;
    }

    var dep_time = departure.get("time");
    if(dep_time != null) {
      this.unix_dep_time = (Long) dep_time;
      if (this.unix_dep_time > 0)
        this.motis_dep_time = Utils.unix_to_motis_time(this.unix_dep_time);
      else
        this.motis_dep_time = 0;
    }else{
      this.unix_dep_time = 0;
      this.motis_dep_time = 0;
    }

    var stop_occ = stop.get("occupancy");
    if(stop_occ instanceof Long occ) {
      this.inbound_occ = occ;
    }else if(stop_occ instanceof String str) {
      this.inbound_occ = Long.parseLong(str);
    }else{
      this.inbound_occ = Long.MIN_VALUE;
    }
  }

  public LocalDateTime arrDt() {
    return LocalDateTime.ofEpochSecond(unix_arr_time, 0, ZoneOffset.UTC);
  }

  public LocalDateTime depDT() {
    return LocalDateTime.ofEpochSecond(unix_dep_time, 0, ZoneOffset.UTC);
  }

  public final int idx;
  public final long unix_arr_time;
  public final long motis_arr_time;
  public final long unix_dep_time;
  public final long motis_dep_time;
  public final String eva_no;
  public final String name;
  public final long inbound_occ;
}
