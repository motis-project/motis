package org.example;

import org.example.data.AbstractConnection;
import org.example.data.Stop;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class SimilarityComparison {
  static final String DATASET = "Swiss";
  static final String TAR = "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\similarity_vs_mcd_on_tso.csv";
  static final String MCD = "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-routing-timeslottedoccupancy.txt";
  static final String[] TSO = new String[]{
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso20.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso24.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso30.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso32.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso36.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso40.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso45.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso48.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso60.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso64.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso72.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso80.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso90.txt",
    "C:\\Users\\Leon\\git\\motis\\Auswertung\\" + DATASET + "\\SplitResults\\r-hhlr-v100-swiss-ontrip-false-raptor-raptor_gpu-tso96.txt"
  };

  public static void main(String[] args) throws IOException, ParseException {
    var sim = new SimilarityComparison();
    sim.process(Path.of(TAR), Path.of(MCD), TSO);
  }

  void process(Path target, Path mcd, String[] tsos) throws IOException, ParseException {
    var csv_out = new ArrayList<String>();
    csv_out.add(csv_header());

    for (var tso : tsos) {
      var result = compare_query_sets(mcd, Path.of(tso));
      csv_out.addAll(result);
    }

    Files.write(target, csv_out);
  }

  String csv_header() {
    return "dataset,slot_size,id,quality";
  }

  String csv_line(int id, int slot_size, double quality) {
    return DATASET + "," + id + "," + slot_size + "," + quality;
  }

  List<String> compare_query_sets(Path reference, Path comparison) throws IOException, ParseException {
    var ref = Files.readAllLines(reference);
    var com = Files.readAllLines(comparison);

    var slot_size = slot_size_from_path(comparison);

    JSONParser parser = new JSONParser();

    var ret = new ArrayList<String>();
    for (int i = 0; i < ref.size(); i++) {
      var r = fromResult((JSONObject) parser.parse(ref.get(i)), slot_size);
      var c = fromResult((JSONObject) parser.parse(com.get(i)), slot_size);

      var q_set_set = quality_set_vs_set(r, c, this::quality_journey_vs_set);
      ret.add(csv_line(i, slot_size, q_set_set));
    }

    return ret;
  }

  int slot_size_from_path(Path p) {
    var file_name = p.getFileName().toString();
    var slot_size_str = file_name.split("-")[8];
    return Integer.parseInt(slot_size_str.substring(2));
  }

  Collection<Journey> fromResult(JSONObject response, int slot_size) {
    var content = (JSONObject)response.get("content");
    var conns = (JSONArray)content.get("connections");
    var ret = new ArrayList<Journey>();
    for (var conn : conns) {
      var j = fromConnection((JSONObject) conn, slot_size);
      ret.add(j);
    }
    return ret;
  }

  Journey fromConnection(JSONObject conn, int slot_size) {
    var c = new SimilarityConnection(conn);

    var occupany_time = c.tso;
    if (slot_size > 0)
      occupany_time = slot_size * occupany_time;

    return new Journey(c.unix_arr_time, c.tripCount, occupany_time);
  }


  /**
   * set1 <- reference set from MCD
   * set2 <- TSO challenger
   */
  double quality_set_vs_set(Collection<Journey> set1, Collection<Journey> set2, QualityJourneySet fn) {
    var qual_set_set = Double.MIN_VALUE;
    Journey j_max = null; //J_max = max{ Q_VS(V_s2, S_1) }

    var comp_s1 = new ArrayList<>(set1);
    var comp_s2 = new ArrayList<>(set2);

    do{
      //0. J_max = max{ Q_VS(V_s2, S_1) }
      j_max = null;
      var q_max = Double.MIN_VALUE;
      for (var j : comp_s2) {
        var q = fn.quality_journey_set(j, comp_s1);
        if (q > q_max) {
          q_max = q;
          j_max = j;
        }
      }

      //1. Q_SS = Q_SS + Q_VS(j_max, S1) - Q_SS * Q_VS(j_max, S1)
      qual_set_set = qual_set_set + q_max - qual_set_set * q_max;

      //2. S_1 = S_1 + J_max
      // add so that further similar journeys have less impact on quality
      comp_s1.add(j_max);

      //3. S_2 = S_2 \ {J_max}
      comp_s2.remove(j_max);

      //5. terminate if |S_2| = 0
    }while(!comp_s2.isEmpty());

    return qual_set_set;
  }

  /**
   * j <- challenger from TSO
   * set <- Reference from MCD
   */
  double quality_journey_vs_set(Journey j, Collection<Journey> set) {
    //Q_VS(J, Set) = min { Q_VV(j, J_s) }
    var q_vs_min = Double.MAX_VALUE;
    for(var j_s : set) {
      var diff_tt = j_s.norm_travel_time() - j.norm_travel_time();
      var diff_tr = j_s.norm_transfers() - j.norm_transfers();
      var diff_ot = j_s.norm_occupancy_time() - j.norm_occupancy_time();

      var q_vs = Math.sqrt(
        Math.pow(diff_tt, 2)
        + Math.pow(diff_tr, 2)
        + Math.pow(diff_ot, 2)
      );

      //normalize
      q_vs = q_vs / 3.0d;
      if (q_vs < q_vs_min)
        q_vs_min = q_vs;
    }

    return q_vs_min;
  }
}


record Journey(
  long travel_time,
  long transfers,
  long occupancy_time
) {

  double norm_occupancy_time() {
    return this.occupancy_time / 2880.0;
  }

  double norm_transfers() {
    return this.transfers / 7.0;
  }

  double norm_travel_time() {
    return 0d; //TODO
  }
}

@FunctionalInterface
interface QualityJourneySet {
  double quality_journey_set(Journey j, Collection<Journey> set);
}

class SimilarityConnection extends AbstractConnection<Stop> {

  public SimilarityConnection(JSONObject conn) {
    super(conn);
  }

  @Override
  public Stop newStop(int idx, JSONObject st) {
    return new Stop(idx, st);
  }
}
