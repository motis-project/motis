// GENERATED FILE - DO NOT MODIFY
// -> see /tools/protocol for information on how to update this file
import { Station } from "@/api/protocol/motis";

// guesser/StationGuesserRequest.fbs
export interface StationGuesserRequest {
  guess_count: number; // default: 8
  input: string;
}

// guesser/StationGuesserResponse.fbs
export interface StationGuesserResponse {
  guesses: Station[];
}
