package org.example;

import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class SplitBatchResult {
  final static String HHLR_BASE_FOLDER = "C:\\Users\\Leon\\git\\motis\\Auswertung\\";
  final static String DATASET = "Hafas-Pretrip";
  final static String ARRAY_QUERIES_IN = "ArrayQueries";
  final static String ARRAY_RESULTS = "ArrayResults";
  final static String SPLIT_RESULTS = "SplitResults";

  final static int SPLIT_QANTITY = 1000;


  public static void main(String[] args) throws IOException, ParseException {
    var inDir = HHLR_BASE_FOLDER + DATASET + "\\" + ARRAY_RESULTS + "\\";
    var outDir = HHLR_BASE_FOLDER + DATASET + "\\" + SPLIT_RESULTS + "\\";
    var inQueries = HHLR_BASE_FOLDER + DATASET + "\\" + ARRAY_QUERIES_IN + "\\";
    var inFiles = Files.list(Path.of(inDir)).toList();

    var parser = new JSONParser();

    for(var inFile : inFiles) {
      System.out.println("Processing File '" + inFile.toString() + "' ...");

      //1. determine base filename
      var baseName = inFile.getFileName().toString();
      baseName = baseName.substring(0, baseName.lastIndexOf('-'));
      System.out.println("  ... base file name '" + baseName + "'");

      var resultMatch = inFile.getFileName().toString();
      resultMatch = resultMatch.substring(resultMatch.lastIndexOf('-')+1, resultMatch.lastIndexOf('.'));
      var resultArrayIndex = Integer.parseInt(resultMatch);

      System.out.println("  ... found result array index of " + resultArrayIndex);

      //2. find matching query file
      var queryFile = Files.list(Path.of(inQueries))
        .map(p -> p.getFileName().toString())
        .filter(fn -> {
          var extracted = fn.substring(fn.lastIndexOf('-')+1, fn.lastIndexOf('.'));
          return Integer.parseInt(extracted) == resultArrayIndex;
          }).findFirst().orElseThrow();
      System.out.println("  ... found matching query file '" + queryFile + "'");

      var queryLines = Files.readAllLines(Path.of(inQueries + queryFile));
      System.out.println("  ... read all lines of Query File (" + queryLines.size() + ")");

      //3. read all lines
      var lines = Files.readAllLines(inFile);
      System.out.println("  ... read all lines of Result file (" + lines.size() + ")");

      if (queryLines.size() != lines.size()) throw new IllegalStateException("Query and Result File don't match!");

      var lineCount = 0;
      var nextBatchStartsAt = lineCount + SPLIT_QANTITY;
      while(nextBatchStartsAt <= lines.size()) {
        System.out.println("    ... extracting sublist [ " + lineCount + "; " + nextBatchStartsAt + " [");
        var subList = lines.subList(lineCount, nextBatchStartsAt);

        var firstQueryOfSublist = queryLines.get(lineCount);
        var parsedQuery = (JSONObject)parser.parse(firstQueryOfSublist);
        System.out.println("    ... parsed query of line " + lineCount);

        var target =  ((String)((JSONObject)parsedQuery.get("destination")).get("target")).substring(1).toLowerCase();
        var search = ((String)((JSONObject)parsedQuery.get("content")).get("search_type")).toLowerCase();

        //baseName: r-hhlr-v100-swiss-ontrip-false-raptor
        //TZiel:    r-hhlr-v100-swiss-ontrip-false-raptor-raptor_cpu-default
        var splitResultFileName = baseName + "-" + target + "-" + search + ".txt";
        System.out.println("    ... determined split file '" + splitResultFileName + "'");

        Files.write(Path.of(outDir + splitResultFileName), subList);
        System.out.println("    ... wrote sublist to split file!");

        lineCount = nextBatchStartsAt;
        nextBatchStartsAt += SPLIT_QANTITY;
      }

    }
  }
}
