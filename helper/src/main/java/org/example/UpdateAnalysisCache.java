package org.example;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class UpdateAnalysisCache {

  final static String ANALYZE_DIR = SplitBatchResult.HHLR_BASE_FOLDER + SplitBatchResult.DATASET + "\\" + SplitBatchResult.SPLIT_RESULTS + "\\";

  public static void main(String[] args) throws IOException, ParseException {
    var inFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> !p.getFileName().toString().endsWith(".anlz"))
      .toList();

    var cacheFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> p.getFileName().toString().endsWith(".anlz"))
      .toList();

    for (var inFile : inFiles) {
      System.out.println("Processing File '" + inFile + "'");
      //try to find existing cache file
      var found = false;
      for (var cache : cacheFiles) {
        if (cache.getFileName().toString().startsWith(inFile.getFileName().toString())) {
          found = true;
          System.out.println("  ... found cache file.");
          break;
        }
      }

      if (found) {
        System.out.println("  ... skipping.");
        continue;
      }

      System.out.println("  ... analyzing");
      var analysis = new ArrayList<>(analyze(inFile));
      System.out.println("  ... analysis done.");

      var outFileName = inFile.getFileName().toString() + ".anlz";
      var outDir = inFile.getParent() + "/" + outFileName;
      System.out.println("  ... determined cache file name '" + outDir + "'");

      var linesOut = new ArrayList<String>();
      linesOut.add(CacheEntry.getHeader());
      for (var c : analysis)
        linesOut.add(c.toString());

      Files.write(Path.of(outDir), linesOut);
      System.out.println("  ... wrote cache file.");
    }
  }


  static class CacheEntry {

    public CacheEntry(String category, String name, String value) {
      this.category = category;
      this.name = name;
      this.value = value;
    }

    String category;
    String name;
    String value;

    static String getHeader() {
      return "category,name,value";
    }

    @Override
    public String toString() {
      return category + "," + name + "," + value;
    }
  }

  static List<CacheEntry> analyze(Path file) throws IOException, ParseException {
    var lines = Files.readAllLines(file);
    var parser = new JSONParser();

    var values = new HashMap<String, HashMap<String, List<Long>>>();
    var total_count = 0;
    var no_conn = 0;
    var counted = 0;

    for (var line : lines) {
      ++total_count;
      var response = (JSONObject) parser.parse(line);
      var content = (JSONObject) response.get("content");

      var connections = (JSONArray) content.get("connections");
      var statistics = (JSONArray) content.get("statistics");

      if(connections.isEmpty()){
        ++no_conn;
        continue;
      }else{
        ++counted;
      }

      for(var s : statistics) {
        var statCat = (JSONObject)s;
        var cat = values.computeIfAbsent((String) statCat.get("category"), k -> new HashMap<>());
        var entries = (JSONArray)statCat.get("entries");
        var foundRaptorConns = false;
        for(var e : entries) {
          var entry = (JSONObject)e;
          var ent = cat.computeIfAbsent((String)entry.get("name"), k -> new ArrayList<>());
          ent.add((Long) entry.get("value"));

          if(((String)statCat.get("category")).equals("raptor") && ((String)entry.get("name")).equals("raptor_connections")) {
            foundRaptorConns = true;
          }
        }

        if(((String)statCat.get("category")).equals("raptor") && !foundRaptorConns) {
          var ent = cat.computeIfAbsent("raptor_connections", k -> new ArrayList<>());
          ent.add((long)connections.size());
        }
      }
    }

    return calculateCacheEntries(values, total_count, no_conn, counted);
  }

  static List<CacheEntry> calculateCacheEntries(HashMap<String, HashMap<String, List<Long>>> stats, int total, int none, int counted) {
    var values = new ArrayList<CacheEntry>();
    values.add(new CacheEntry("general", "total_count", "" + total));
    values.add(new CacheEntry("general", "no_conn", "" + none));
    values.add(new CacheEntry("general", "counted", "" + counted));

    for(var cat : stats.entrySet()) {
      var catName = cat.getKey();

      for(var val : cat.getValue().entrySet()) {
        var valName = val.getKey();

        //1. sum
        var sum = val.getValue().stream().reduce(0L, Long::sum);
        values.add(new CacheEntry(catName, "sum_" + valName, "" + sum));

        //2. avg
        values.add(new CacheEntry(catName, "avg_" + valName, "" + (sum / (0f + val.getValue().size()))));

        var sorted = val.getValue();
        sorted.sort(Long::compare);

        //quantiles
        values.add(new CacheEntry(catName, "q99_" + valName, "" + quantile(sorted, 0.99)));
        values.add(new CacheEntry(catName, "q90_" + valName, "" + quantile(sorted, 0.90)));
        values.add(new CacheEntry(catName, "q80_" + valName, "" + quantile(sorted, 0.80)));
        values.add(new CacheEntry(catName, "q50_" + valName, "" + quantile(sorted, 0.50)));
      }
    }

    return values;
  }

  static long quantile(List<Long> sorted, double quantile) {
    if (quantile == 1.0) {
      return sorted.get(sorted.size() - 1);
    } else {
      var idx = Math.min(Math.round(quantile * (sorted.size() - 1)), sorted.size() - 1);
      return sorted.get((int) idx);
    }
  }
}
