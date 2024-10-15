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
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ResultComparator {

  static class CompareTrip {
    public CompareTrip(JSONObject trip, List<Stop> stops) {
      var range = (JSONObject) trip.get("range");

      long fromIdx = (Long) range.get("from");
      long toIdx = (Long) range.get("to");

      this.from = stops.get((int) fromIdx);
      this.to = stops.get((int) toIdx);
    }

    final Stop from;
    final Stop to;
  }

  static class CompareConnection extends AbstractConnection<Stop> {
    public final long query_id;
    List<CompareTrip> trips;

    @Override
    public Stop newStop(int idx, JSONObject st) {
      return new Stop(idx, st);
    }

    public CompareConnection(long id, JSONObject connection) {
      super(connection);
      this.query_id = id;

      this.trips = new ArrayList<>();
      var trips = (JSONArray) connection.get("trips");
      for (var t : trips) {
        var tr = (JSONObject) t;
        var curr = new CompareTrip(tr, super.stops);
        this.trips.add(curr);
      }
    }

    public String toString() {
      return departureFmt() + "\t\t" + arrivalFmt() + "\t\t" + durationStr + "\tTR: " + tripCount + "\tMOC: " + moc + "\tTSO: " + tso + "\tMTC: " + mtc;
    }

    public boolean dominates(CompareConnection toDominate) {
      if (toDominate == null) return true;

      return tripCount <= toDominate.tripCount && moc <= toDominate.moc && tso <= toDominate.tso && mtc <= toDominate.mtc && unix_arr_time <= toDominate.unix_arr_time;
    }

    public int compareTo(CompareConnection o) {
      if (o == null) return -1;

      var tc = Long.compare(this.tripCount, o.tripCount);
      if (tc != 0) return tc;

      var mc = Long.compare(this.moc, o.moc);
      if (mc != 0) return mc;

      var tso = Long.compare(this.tso, o.tso);
      if (tso != 0) return tso;

      var mtc = Long.compare(this.mtc, o.mtc);
      if (mtc != 0) return mtc;

      //var dc = Long.compare(this.duration.getSeconds(), o.duration.getSeconds());
      //if (dc != 0) return dc;

      return Long.compare(this.unix_arr_time, o.unix_arr_time);
    }

    public String getChangeWaitingTime() {
      var bld = new StringBuilder("Connection with TR: ")
        .append(tripCount).append(";\tMOC: ").append(moc).append("\n");
      for (int i = 1; i < tripCount; i++) {
        var duration = Duration.between(trips.get(i - 1).to.arrDt(), trips.get(i).from.depDT());
        bld.append(i).append(" change: ").append(String.format("%d:%02d:%02d", duration.toHours(), duration.toMinutesPart(), duration.toSecondsPart()));
        bld.append(";\t");
      }
      return bld.toString();
    }
  }

  static class ComparisonResult {

    public final long id;

    List<CompareConnection> raptorConns;
    List<CompareConnection> routingConns;

    int matchingConns;
    int rpcConCnt = 0;
    int rocConCnt = 0;

    int rpcTrMocMask = 0;
    int rocTrMocMask = 0;

    public ComparisonResult(long id) {
      this.raptorConns = new ArrayList<>();
      this.routingConns = new ArrayList<>();
      this.id = id;
      this.matchingConns = 0;
    }

    void addMatch(CompareConnection rpc, CompareConnection roc) {
      raptorConns.add(rpc);
      routingConns.add(roc);
      ++rocConCnt;
      ++rpcConCnt;
      ++matchingConns;
      rpcTrMocMask = _update_mask(rpcTrMocMask, rpc);
      rocTrMocMask = _update_mask(rocTrMocMask, roc);
    }

    void addRpcOnly(CompareConnection rpc) {
      raptorConns.add(rpc);
      routingConns.add(null);
      ++rpcConCnt;
      rpcTrMocMask = _update_mask(rpcTrMocMask, rpc);
    }

    void addRocOnly(CompareConnection roc) {
      raptorConns.add(null);
      routingConns.add(roc);
      ++rocConCnt;
      rocTrMocMask = _update_mask(rocTrMocMask, roc);
    }

    int _update_mask(int mask, CompareConnection c) {
      final var tr = c.tripCount;
      final var moc = c.moc;

      final var combined = (1 << (tr * (moc + 1)));
      mask = mask | combined;
      return mask;
    }

    boolean isFullMatch() {
      return matchingConns == Math.max(raptorConns.size(), routingConns.size());
    }

    boolean isFullMatchOnTripsAndMoc() {
      return rpcTrMocMask == rocTrMocMask;
    }

    public String toString() {
      if (routingConns.size() != raptorConns.size()) throw new IllegalStateException("Mismatch conn count!");
      var empty = String.format("%-88s", "---");
      var bld = new StringBuilder();
      bld.append("Comparison for Query ID: ").append(id).append(";\tFound full match: ").append(isFullMatch()).append(";\tMatching Connection Count: ").append((rpcConCnt == rocConCnt)).append("\n");
      bld.append("==================================================================================================\n");
      for (int i = 0; i < raptorConns.size(); i++) {
        var lhs = raptorConns.get(i);
        var rhs = routingConns.get(i);

        var lhsString = lhs != null ? lhs.toString() : empty;
        var rhsString = rhs != null ? rhs.toString() : empty;
        var matches = (lhs != null && rhs != null && lhs.compareTo(rhs) == 0) ? "M" : "-";

        bld.append(String.format("%02d", i)).append(": ").append(lhsString).append("\t\t").append(matches).append("\t\t").append(rhsString).append("\n");
      }

      return bld.toString();
    }
  }

  static List<CompareConnection> getConnections(long id, JSONObject content) {
    var conns = (JSONArray) content.get("connections");
    var cs = new ArrayList<CompareConnection>();
    for (var c : conns) {
      var connection = (JSONObject) c;
      cs.add(new CompareConnection(id, connection));
    }
    cs.sort((lhs, rhs) -> {
      if (lhs == null || rhs == null) throw new IllegalStateException("Received a null!");
      return lhs.compareTo(rhs);
    });
    return cs;
  }

  static HashMap<Long, List<CompareConnection>> transform(List<String> lines) throws ParseException {
    var parser = new JSONParser();
    var map = new HashMap<Long, List<CompareConnection>>();
    for (var line : lines) {
      var response = (JSONObject) parser.parse(line);
      var id = (Long) response.get("id");
      var conns = getConnections(id, (JSONObject) response.get("content"));
      map.put(id, conns);
    }
    return map;
  }

  static List<ComparisonResult> compare(long resCount, Map<Long, List<CompareConnection>> raptorConns, Map<Long, List<CompareConnection>> routingConns, int up_to_line) {

    var r = new ArrayList<ComparisonResult>();
    for (long queryId = 1; queryId <= resCount && queryId < up_to_line; queryId++) {

      var rpc = raptorConns.get(queryId);
      var roc = routingConns.get(queryId);

      var result = new ComparisonResult(queryId);

      var rpcIdx = 0;
      var rocIdx = 0;

      while (rpcIdx < rpc.size() && rocIdx < roc.size()) {

        var rapc = rpc.get(rpcIdx);
        var rouc = roc.get(rocIdx);

        var compare = rapc.compareTo(rouc);
        if (compare < 0) {
          // LHS better than RHS
          result.addRpcOnly(rapc);
          ++rpcIdx;
        } else if (compare > 0) {
          // RHS better than LHS
          result.addRocOnly(rouc);
          ++rocIdx;
        } else {
          // match
          result.addMatch(rapc, rouc);
          ++rpcIdx;
          ++rocIdx;
        }
      }

      while (rpcIdx < rpc.size()) {
        result.addRpcOnly(rpc.get(rpcIdx));
        ++rpcIdx;
      }

      while (rocIdx < roc.size()) {
        result.addRocOnly(roc.get(rocIdx));
        ++rocIdx;
      }

      r.add(result);
    }

    return r;
  }

  static void printChangeWaitingTimes(ComparisonResult compare) {
    System.out.println("Query ID " + compare.id);
    System.out.println("======================================================================================");
    for (int cIdx = 0; cIdx < compare.raptorConns.size(); cIdx++) {
      var rpc = compare.raptorConns.get(cIdx);
      var roc = compare.routingConns.get(cIdx);
      if (rpc != null && roc == null) {
        System.out.println(rpc.getChangeWaitingTime());
        System.out.println();
      }
    }
  }

  static void filterDominated(List<CompareConnection> conns) {
    for (int i = 0; i < conns.size(); i++) {
      var dominator = conns.get(i);
      for (int j = i + 1; j < conns.size(); j++) {
        var toDominate = conns.get(j);
        if (dominator.dominates(toDominate)) {
          conns.remove(j--);
        }
      }
    }
  }

  static final boolean fullPrint = false;

  public static void main(String[] args) throws IOException, ParseException {
    System.out.print("Reading Files ...");
    var raptorLines = Files.readAllLines(Path.of("verification/sbb-small/r-raptor_cpu-mtc.txt"));
    var routingLines = Files.readAllLines(Path.of("verification/sbb-small/r-raptor_gpu-mtc-as-refactor.txt"));

    //if (raptorLines.size() != routingLines.size()) throw new IllegalStateException("Line Counts don't match!");
    System.out.println("Ok");

    System.out.print("Parsing data ...");
    var raptorConns = transform(raptorLines);
    var routingConns = transform(routingLines);
    //for(var e : routingConns.entrySet()) {
    //  filterDominated(e.getValue(), mocRelevant);
    //}
    System.out.println("Ok");

    System.out.print("Comparing ...");
    var comparison = compare(raptorLines.size(), raptorConns, routingConns, 1001);
    System.out.println("Ok");

    var full_match_count = 0;
    var matchingConCount = 0;
    var totalMatchingCnt = 0;
    var totalConnCnt = 0;
    var matchingTrMoc = 0;
    var moreRpcConns = new ArrayList<ComparisonResult>();
    var moreRocConns = new ArrayList<ComparisonResult>();
    var totalCount = raptorLines.size();

    for (var res : comparison) {
      if (fullPrint || !res.isFullMatch()) {
        System.out.println(res.toString());
        System.out.println();
        System.out.println();
      }

      totalMatchingCnt += res.matchingConns;
      totalConnCnt += res.raptorConns.size();

      if (res.isFullMatch())
        ++full_match_count;

      if (res.isFullMatchOnTripsAndMoc())
        ++matchingTrMoc;

      if (res.rocConCnt == res.rpcConCnt)
        ++matchingConCount;

      if (res.rpcConCnt > res.rocConCnt)
        moreRpcConns.add(res);

      if (res.rocConCnt > res.rpcConCnt)
        moreRocConns.add(res);
    }

    System.out.println("Statistics:");
    System.out.println(String.format("%35s", "Full Matches: ") + "\t" + String.format("%4d", full_match_count) + "/" + totalCount + ";\t" + String.format("%.2f", (full_match_count + 0.0) / totalCount));
    System.out.println(String.format("%35s", "ConnCnt Matches (TR,MOC): ") + "\t" + String.format("%4d", matchingTrMoc) + "/" + totalCount + ";\t" + String.format("%.2f", (matchingTrMoc + 0.0) / totalCount));
    System.out.println(String.format("%35s", "Total Conn. Matches: ") + "\t" + String.format("%4d", totalMatchingCnt) + "/" + totalConnCnt + ";\t" + String.format("%.2f", (totalMatchingCnt + 0.0) / totalConnCnt));
    System.out.println();
    System.out.println(String.format("%35s", "Connection Count Matches: ") + "\t" + String.format("%4d", matchingConCount) + "/" + totalCount + ";\t" + String.format("%.2f", (matchingConCount + 0.0) / totalCount));
    System.out.println();
    ;
    System.out.println(String.format("%35s", "More Raptor Conns: ") + "\t" + String.format("%4d", moreRpcConns.size()) + "/" + totalCount);
    System.out.println(String.format("%35s", "More Routing Conns: ") + "\t" + String.format("%4d", moreRocConns.size()) + "/" + totalCount);
    System.out.println();
    System.out.println();

    if (!moreRpcConns.isEmpty()) {
      var blcRpc = new StringBuilder("More Raptor Connections for Queries: ");
      moreRpcConns.forEach((e) -> blcRpc.append(e.id).append(", "));
      System.out.println(blcRpc);
      System.out.println();
      System.out.println();
    }

    if (!moreRocConns.isEmpty()) {
      var blcRoc = new StringBuilder("More Routing Connections for Queries: ");
      moreRocConns.forEach((e) -> blcRoc.append(e.id).append(", "));
      System.out.println(blcRoc);
    }
  }

}
