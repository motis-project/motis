package org.example;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class UpdateCsvFromAnalysiscache {

  final static String BASE_PATH = SplitBatchResult.HHLR_BASE_FOLDER + SplitBatchResult.DATASET + "\\";
  final static String SPLIT_RESULT = SplitBatchResult.SPLIT_RESULTS;

  final static String CSV_FILE = "analysis.csv";
  final static String FULL_CSV_FILE = "full-analysis.csv";

  public static void main(String[] args) throws IOException {
    var files = Files.list(Path.of(BASE_PATH + SPLIT_RESULT))
      .filter(p -> p.getFileName().toString().endsWith(".anlz"))
      .toList();

    var fullFiles = Files.list(Path.of(BASE_PATH + SPLIT_RESULT))
      .filter(p -> p.getFileName().toString().endsWith(".anlz_full"))
      .toList();

    var csvPath = BASE_PATH + CSV_FILE;
    var fullCsvPath = BASE_PATH + FULL_CSV_FILE;
    var csv = Files.newBufferedWriter(Path.of(csvPath));
    var fullCsv = Files.newBufferedWriter(Path.of(fullCsvPath));

    AnalysisCache.writeCsvHeader(csv);
    AnalysisCache.writeCsvHeader(fullCsv);
    for(var cacheFile : files) {
      var results = new AnalysisCache(cacheFile);
      results.writeCsvLines(csv);
    }
    csv.close();

    for(var fullCacheFile : fullFiles) {
      var results = new AnalysisCache(fullCacheFile);
      results.writeCsvLines(fullCsv);
    }
    fullCsv.close();
  }

  static class AnalysisCache {
    String gpu;
    String dataset;
    String queryType;
    boolean largeStations;
    String searchType;
    String target;
    List<Statistic> stats;


    AnalysisCache(Path file) throws IOException {
      var fileName = file.getFileName().toString();
      fileName = fileName.substring(0, fileName.indexOf('.'));
      var configComponents = fileName.split("-");

      gpu = configComponents[2];
      dataset = configComponents[3];
      queryType = configComponents[4];
      largeStations = Boolean.parseBoolean(configComponents[5]);
      target = configComponents[6];
      searchType = configComponents[7];
      stats = new ArrayList<>();

      var lines = Files.readAllLines(file);
      for(var line : lines) {
        if(line.startsWith("category")) continue; //skip header

        var split = line.split(",");
        stats.add(new Statistic(split[0], split[1], split[2], split[3]));
      }
    }

    void writeCsvLines(BufferedWriter out) throws IOException {
      var line_begin = gpu + "," + dataset + "," + queryType + "," + largeStations + "," + searchType + "," + target + ",";
      for(var s : stats) {
        var line_end = s.category + "," + s.name + "," + s.id + "," + s.val;
        out.write(line_begin + line_end + "\n");
      }
    }

    static void writeCsvHeader(BufferedWriter out) throws IOException {
      out.write("gpu,dataset,query_type,large_stations,search_type,target,stat_category,stat_name,id,val\n");
    }
  }

  static class Statistic {
    public Statistic(String category, String name, String id, String val) {
      this.category = category;
      this.name = name;
      this.val = val;
      this.id = id;
    }

    String category;
    String name;
    String id;
    String val;
  }
}
