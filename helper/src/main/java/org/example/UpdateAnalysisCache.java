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
import java.util.function.Function;

public class UpdateAnalysisCache {

  final static String ANALYZE_DIR = SplitBatchResult.HHLR_BASE_FOLDER + SplitBatchResult.DATASET + "\\" + SplitBatchResult.SPLIT_RESULTS + "\\";

  public static void main(String[] args) throws IOException, ParseException {
    var inFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> !p.getFileName().toString().endsWith(".anlz") && !p.getFileName().toString().endsWith(".anlz_full"))
      .toList();

    var cacheFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> p.getFileName().toString().endsWith(".anlz"))
      .toList();

    var fullCacheFiles = Files.list(Path.of(ANALYZE_DIR))
      .filter(p -> p.getFileName().toString().endsWith(".anlz_full"))
      .toList();

    for (var inFile : inFiles) {
      System.out.println("Processing File '" + inFile + "'");
      //try to find existing cache file
      var foundCacheFile = false;
      for (var cache : cacheFiles) {
        if (cache.getFileName().toString().startsWith(inFile.getFileName().toString())) {
          foundCacheFile = true;
          System.out.println("  ... found cache file.");
          break;
        }
      }

      var foundFullCacheFile = false;
      for (var cache : fullCacheFiles) {
        if (cache.getFileName().toString().startsWith(inFile.getFileName().toString())) {
          foundFullCacheFile = true;
          System.out.println("  ... found full cache file.");
          break;
        }
      }

      if (foundCacheFile && foundFullCacheFile) {
        System.out.println("  ... skipping.");
        continue;
      }

      System.out.println("  ... analyzing");
      var cacheEntries = new ArrayList<CacheEntry>();
      var fullCacheEntries = new ArrayList<CacheEntry>();
      analyze(inFile, !foundCacheFile, cacheEntries::addAll, !foundFullCacheFile, fullCacheEntries::addAll);
      System.out.println("  ... analysis done.");

      if (!cacheEntries.isEmpty()) {
        var outFileName = inFile.getFileName().toString() + ".anlz";
        var outDir = inFile.getParent() + "/" + outFileName;
        writeCacheFile(outDir, cacheEntries);
      }

      if (!fullCacheEntries.isEmpty()) {
        var outFileName = inFile.getFileName().toString() + ".anlz_full";
        var outDir = inFile.getParent() + "/" + outFileName;
        writeCacheFile(outDir, fullCacheEntries);
      }
    }
  }

  static class CacheEntry {

    public CacheEntry(String category, String name, String id, String value) {
      this.category = category;
      this.name = name;
      this.value = value;
      this.id = id;
    }

    String category;
    String name;
    String value;
    String id;

    static String getHeader() {
      return "category,name,id,value";
    }

    @Override
    public String toString() {
      return category + "," + name + "," + id + "," + value;
    }
  }

  static void writeCacheFile(String outDir, List<CacheEntry> cacheEntries) throws IOException {
    System.out.println("  ... determined cache file name '" + outDir + "'");

    var linesOut = new ArrayList<String>();
    linesOut.add(CacheEntry.getHeader());
    for (var c : cacheEntries)
      linesOut.add(c.toString());

    Files.write(Path.of(outDir), linesOut);
    System.out.println("  ... wrote cache file.");
  }

  static void analyze(Path file, boolean doCacheUpdate, Function<List<CacheEntry>, Boolean> cacheCb,
                      boolean doFullCacheUpdate, Function<List<CacheEntry>, Boolean> fullCacheCb) throws IOException, ParseException {
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

      if (connections.isEmpty()) {
        ++no_conn;
        continue;
      } else {
        ++counted;
      }

      for (var s : statistics) {
        var statCat = (JSONObject) s;
        var cat = values.computeIfAbsent((String) statCat.get("category"), k -> new HashMap<>());
        var entries = (JSONArray) statCat.get("entries");
        var foundRaptorConns = false;
        for (var e : entries) {
          var entry = (JSONObject) e;
          var ent = cat.computeIfAbsent((String) entry.get("name"), k -> new ArrayList<>());
          ent.add((Long) entry.get("value"));

          if (((String) statCat.get("category")).equals("raptor") && ((String) entry.get("name")).equals("raptor_connections")) {
            foundRaptorConns = true;
          }
        }

        if (((String) statCat.get("category")).equals("raptor") && !foundRaptorConns) {
          var ent = cat.computeIfAbsent("raptor_connections", k -> new ArrayList<>());
          ent.add((long) connections.size());
        }

        if(((String)statCat.get("category")).equals("routing")){
          var ent = cat.computeIfAbsent("raptor_connections", k -> new ArrayList<>());
          ent.add((long)connections.size());
        }
      }
    }

    if (doCacheUpdate) {
      cacheCb.apply(calculateCacheEntries(values, total_count, no_conn, counted));
    }

    if (doFullCacheUpdate) {
      fullCacheCb.apply(calculateFullCacheEntries(values));
    }

  }

  static List<CacheEntry> calculateCacheEntries(HashMap<String, HashMap<String, List<Long>>> stats, int total, int none, int counted) {
    var values = new ArrayList<CacheEntry>();
    values.add(new CacheEntry("general", "total_count", "" + 0, "" + total));
    values.add(new CacheEntry("general", "no_conn", "" + 0, "" + none));
    values.add(new CacheEntry("general", "counted", "" + 0, "" + counted));

    for (var cat : stats.entrySet()) {
      var catName = cat.getKey();

      for (var val : cat.getValue().entrySet()) {
        var valName = val.getKey();

        //1. sum
        var sum = val.getValue().stream().reduce(0L, Long::sum);
        values.add(new CacheEntry(catName, "sum_" + valName, "" + 0, "" + sum));

        //2. avg
        values.add(new CacheEntry(catName, "avg_" + valName, "" + 0, "" + (sum / (0f + val.getValue().size()))));

        var sorted = val.getValue();
        sorted.sort(Long::compare);

        //quantiles
        values.add(new CacheEntry(catName, "q99_" + valName, "" + 0, "" + quantile(sorted, 0.99)));
        values.add(new CacheEntry(catName, "q90_" + valName, "" + 0, "" + quantile(sorted, 0.90)));
        values.add(new CacheEntry(catName, "q80_" + valName, "" + 0, "" + quantile(sorted, 0.80)));
        values.add(new CacheEntry(catName, "q75_" + valName, "" + 0, "" + quantile(sorted, 0.75)));
        values.add(new CacheEntry(catName, "q50_" + valName, "" + 0, "" + quantile(sorted, 0.50)));
        values.add(new CacheEntry(catName, "q25_" + valName, "" + 0, "" + quantile(sorted, 0.25)));
        values.add(new CacheEntry(catName, "min_" + valName, "" + 0, "" + sorted.get(0)));
        values.add(new CacheEntry(catName, "max_" + valName, "" + 0, "" + quantile(sorted, 1)));
      }
    }

    return values;
  }

  static List<CacheEntry> calculateFullCacheEntries(HashMap<String, HashMap<String, List<Long>>> stats) {
    var values = new ArrayList<CacheEntry>();
    for (var cat : stats.entrySet()) {
      var catName = cat.getKey();

      for (var val : cat.getValue().entrySet()) {
        var valName = val.getKey();

        var count = 0;
        for(var v : val.getValue()) {
          values.add(new CacheEntry(catName, valName, "" + count, "" + v));
          ++count;
        }
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
